from tools import toolbox
print([s["function"]["name"] for s in toolbox.schemas])        # ['calculator', 'get_weather']
print(toolbox.call("calculator", {"operation": "*", "num1": 24, "num2": 6}))   # 144.0
print(toolbox.call("get_weather", {"city": "Kochi"}))