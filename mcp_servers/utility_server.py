# mcp_servers/utility_server.py
import sys
from typing import Literal
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("utilities")

@mcp.tool()
def calculator(operation: Literal["+", "-", "*", "/"], num1: float, num2: float) -> str:
    """Perform basic arithmetic: addition, subtraction, multiplication or division."""
    if operation == "+": return str(num1 + num2)
    if operation == "-": return str(num1 - num2)
    if operation == "*": return str(num1 * num2)
    if num2 == 0: return "Error: cannot divide by zero"
    return str(num1 / num2)

@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current temperature, humidity and wind speed for a city."""
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={"name": city, "count": 1, "language": "en", "format": "json"},
                       timeout=10).json()
    if not geo.get("results"):
        return {"error": f"Could not find a city called '{city}'"}
    loc = geo["results"][0]
    cur = requests.get("https://api.open-meteo.com/v1/forecast",
                       params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                               "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"},
                       timeout=10).json()["current"]
    return {"city": loc["name"], "country": loc.get("country"),
            "temperature": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"],
            "wind_speed": cur["wind_speed_10m"]}

if __name__ == "__main__":
    mcp.run(transport="stdio")