import json
import mpmath
import os


def stringify(data):
    if data is None:
        return ''
    if not isinstance(data, (list, dict)):
        return str(data)

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                continue
            if isinstance(v, (int, float)):
                data[k] = str(v)
            else:
                data = stringify(data[k])
        return data
    if isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, str):
                continue
            if isinstance(v, (int, float)):
                data[i] = str(v)
            else:
                data = stringify(data[i])
        return data


def serialize_value(value):
    """Serialize a value to a JSON-compatible format."""
    if isinstance(value, dict):
        # Recursively handle dictionaries (convert keys to string and values)
        return {str(key): serialize_value(val) for key, val in value.items()}
    elif isinstance(value, list):
        # Recursively handle lists
        return [serialize_value(item) for item in value]
    elif isinstance(value, set):
        # Handle sets (convert to list for JSON compatibility)
        return list(serialize_value(item) for item in value)
    elif isinstance(value, tuple):
        # Handle tuples (convert to list)
        return [serialize_value(item) for item in value]
    else:
        # Handle other types (e.g., ints, floats, strings)
        return str(value)


def deserialize_value(value):
    """Deserialize a value back to its original Python format."""
    if isinstance(value, dict):
        # Recursively handle dictionaries
        return {key: deserialize_value(val) for key, val in value.items()}
    elif isinstance(value, list):
        # Recursively handle lists
        return [deserialize_value(item) for item in value]
    else:
        # Return the value as-is (since it's a basic type)
        if value in ('True', 'False'):
            return bool(value)
        if value == '+inf':
            return mpmath.inf
        if value == '-inf':
            return mpmath.ninf
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except:
            return value


def to_json(data: dict, file_name: str = 'data'):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(serialize_value(data), file, indent=4)


def from_json(file_name: str = 'data'):
    if os.path.exists(f'{file_name}.json'):
        with open(f'{file_name}.json', 'r') as file:
            data = json.load(file)
        return deserialize_value(data['data'])
    return None


def unpack_json(json_data):
    formatted = {}

    for d in json_data:
        key = (*d['start'], *d['trajectory'])
        del d['start']
        del d['trajectory']
        formatted[key] = d
    return formatted
