import json


def write_to_json(location: str, key: str, value: str) -> None:
    with open(location, 'r') as f:
        data = json.load(f)
    data[key] = value