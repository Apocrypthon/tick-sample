import time
import unittest
from unittest.mock import patch

from src.subagents.executor import (
    SubagentExecutor,
    SubagentStatus,
    SubagentResult,
    _background_tasks,
    _background_tasks_lock,
    get_background_task_result,
    list_background_tasks,
    cleanup_background_task,
)
from src.subagents.config import SubagentConfig


class TestSubagentExecutorBackgroundTasks(unittest.TestCase):
    def setUp(self):
        # Clear background tasks before each test
        with _background_tasks_lock:
            _background_tasks.clear()

        self.config = SubagentConfig(
            name="test_agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            timeout_seconds=1,  # short timeout for tests
        )
        self.executor = SubagentExecutor(
            config=self.config,
            tools=[],
        )

    def wait_for_terminal_status(self, task_id, timeout=2.0):
        """Helper to wait for a task to reach a terminal state."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = get_background_task_result(task_id)
            if result and result.status in {
                SubagentStatus.COMPLETED,
                SubagentStatus.FAILED,
                SubagentStatus.TIMED_OUT,
            }:
                return result
            time.sleep(0.05)
        self.fail(
            f"Task {task_id} did not reach terminal status within {timeout} seconds"
        )

    @patch("src.subagents.executor.SubagentExecutor.execute")
    def test_execute_async_success(self, mock_execute):
        # Setup mock to return a completed result
        expected_result = SubagentResult(
            task_id="dummy",
            trace_id="dummy_trace",
            status=SubagentStatus.COMPLETED,
            result="Success!",
            ai_messages=[{"content": "Success!"}],
        )
        mock_execute.return_value = expected_result

        # Execute
        task_id = self.executor.execute_async("do something")

        # Verify initial state
        initial_result = get_background_task_result(task_id)
        self.assertIsNotNone(initial_result)
        # It could be PENDING or RUNNING depending on how fast the thread starts,
        # but it shouldn't be COMPLETED immediately.
        self.assertIn(
            initial_result.status,
            {SubagentStatus.PENDING, SubagentStatus.RUNNING, SubagentStatus.COMPLETED},
        )

        # Wait for completion
        final_result = self.wait_for_terminal_status(task_id)

        # Verify final state
        self.assertEqual(final_result.status, SubagentStatus.COMPLETED)
        self.assertEqual(final_result.result, "Success!")
        self.assertEqual(final_result.ai_messages, [{"content": "Success!"}])
        self.assertIsNotNone(final_result.completed_at)
        self.assertIsNotNone(final_result.started_at)
        self.assertEqual(final_result.error, None)

        mock_execute.assert_called_once()
        # Check args passed to execute - it should be the task string and a result_holder
        args, kwargs = mock_execute.call_args
        self.assertEqual(args[0], "do something")
        self.assertIsInstance(args[1], SubagentResult)
        self.assertEqual(args[1].task_id, task_id)

    @patch("src.subagents.executor.SubagentExecutor.execute")
    def test_execute_async_timeout(self, mock_execute):
        # Setup test with a very short timeout
        self.executor.config.timeout_seconds = 0.1

        # Setup mock to sleep longer than the timeout
        def slow_execute(*args, **kwargs):
            time.sleep(0.3)
            return SubagentResult(
                task_id="dummy",
                trace_id="dummy_trace",
                status=SubagentStatus.COMPLETED,
            )

        mock_execute.side_effect = slow_execute

        # Execute
        task_id = self.executor.execute_async("do something slow")

        # Wait for completion
        final_result = self.wait_for_terminal_status(task_id)

        # Verify it timed out
        self.assertEqual(final_result.status, SubagentStatus.TIMED_OUT)
        self.assertIn("timed out after", final_result.error)
        self.assertIsNotNone(final_result.completed_at)

    @patch("src.subagents.executor._execution_pool.submit")
    def test_execute_async_exception(self, mock_submit):
        # Setup mock to raise an unexpected exception when submitting to the pool
        mock_submit.side_effect = RuntimeError("Pool submission failed")

        # Execute
        task_id = self.executor.execute_async("do something error-prone")

        # Wait for completion
        final_result = self.wait_for_terminal_status(task_id)

        # Verify it failed
        self.assertEqual(final_result.status, SubagentStatus.FAILED)
        self.assertIn("Pool submission failed", final_result.error)
        self.assertIsNotNone(final_result.completed_at)

    def test_get_background_task_result(self):
        # Setup a dummy task
        task_id = "test_task_123"
        result = SubagentResult(
            task_id=task_id, trace_id="trace", status=SubagentStatus.PENDING
        )
        with _background_tasks_lock:
            _background_tasks[task_id] = result

        # Retrieve it
        retrieved = get_background_task_result(task_id)
        self.assertEqual(retrieved, result)

        # Retrieve unknown
        self.assertIsNone(get_background_task_result("unknown_task"))

    def test_list_background_tasks(self):
        # Setup dummy tasks
        task1 = SubagentResult(
            task_id="t1", trace_id="trace1", status=SubagentStatus.PENDING
        )
        task2 = SubagentResult(
            task_id="t2", trace_id="trace2", status=SubagentStatus.RUNNING
        )
        with _background_tasks_lock:
            _background_tasks["t1"] = task1
            _background_tasks["t2"] = task2

        # List them
        tasks = list_background_tasks()
        self.assertEqual(len(tasks), 2)
        self.assertIn(task1, tasks)
        self.assertIn(task2, tasks)

    def test_cleanup_background_task(self):
        # Setup dummy tasks in various states
        task_pending = SubagentResult(
            task_id="t_pending", trace_id="trace", status=SubagentStatus.PENDING
        )
        task_running = SubagentResult(
            task_id="t_running", trace_id="trace", status=SubagentStatus.RUNNING
        )
        task_completed = SubagentResult(
            task_id="t_completed", trace_id="trace", status=SubagentStatus.COMPLETED
        )
        task_failed = SubagentResult(
            task_id="t_failed", trace_id="trace", status=SubagentStatus.FAILED
        )
        task_timeout = SubagentResult(
            task_id="t_timeout", trace_id="trace", status=SubagentStatus.TIMED_OUT
        )

        with _background_tasks_lock:
            _background_tasks["t_pending"] = task_pending
            _background_tasks["t_running"] = task_running
            _background_tasks["t_completed"] = task_completed
            _background_tasks["t_failed"] = task_failed
            _background_tasks["t_timeout"] = task_timeout

        # Attempt to clean up all
        cleanup_background_task("t_pending")
        cleanup_background_task("t_running")
        cleanup_background_task("t_completed")
        cleanup_background_task("t_failed")
        cleanup_background_task("t_timeout")
        cleanup_background_task("t_unknown")  # Should do nothing safely

        # Verify only non-terminal tasks remain
        with _background_tasks_lock:
            self.assertIn("t_pending", _background_tasks)
            self.assertIn("t_running", _background_tasks)
            self.assertNotIn("t_completed", _background_tasks)
            self.assertNotIn("t_failed", _background_tasks)
            self.assertNotIn("t_timeout", _background_tasks)
