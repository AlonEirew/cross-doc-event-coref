import json


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path) as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print('trying to read file in blocks')
        with open(file_path) as big_file:
            json_string = ''
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)
