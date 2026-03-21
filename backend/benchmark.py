import timeit
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.config.app_config import AppConfig
from src.config.tool_config import ToolConfig
from src.config.model_config import ModelConfig

def run_benchmark():
    # Setup some test config with many tools and models
    tools = [ToolConfig(name=f"tool_{i}", type="mcp", params={}, group="g", use="true") for i in range(1000)]
    models = [ModelConfig(name=f"model_{i}", provider="openai", model="gpt-4", use="true") for i in range(1000)]

    config = AppConfig(
        tools=tools,
        models=models,
        sandbox={"type": "docker", "timeout": 30, "use": "true"}
    )

    target_tool = "tool_999"
    target_model = "model_999"

    def search_tool():
        return config.get_tool_config(target_tool)

    def search_model():
        return config.get_model_config(target_model)

    def not_found_tool():
        return config.get_tool_config("not_found")

    tool_time = timeit.timeit(search_tool, number=10000)
    model_time = timeit.timeit(search_model, number=10000)
    not_found_time = timeit.timeit(not_found_tool, number=10000)

    print(f"Tool search (last item): {tool_time:.5f}s for 10000 loops")
    print(f"Model search (last item): {model_time:.5f}s for 10000 loops")
    print(f"Not found search: {not_found_time:.5f}s for 10000 loops")

if __name__ == "__main__":
    run_benchmark()
