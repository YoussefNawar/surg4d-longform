spec_node_distances_through_time = {
    "type": "function",
    "function": {
        "name": "node_distances_through_time",
        "description": "Returns the distances at all timesteps for two nodes",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id_1": {"type": "integer", "description": "The first node's id"},
                "node_id_2": {"type": "integer", "description": "The second node's id"},
            },
            "required": ["node_id_1", "node_id_2"],
        },
    },
}


def node_distances_through_time(node_id_1: int, node_id_2: int):
    return [{"timestep": i, "distance": 0.5 * i} for i in range(20)]

spec_weather_munich = {
    "type": "function",
    "function": {
        "name": "weather_munich",
        "description": "Returns the weather in Munich for a given date",
        "parameters": {"date": "string (YYYY-MM-DD)"},
        "required": ["date"],
    },
}

def weather_munich(date: str):
    return f"Mocked weather for Munich on {date}: Sunny, 25°C."

ALL_TOOLS = [spec_node_distances_through_time]
