import json

def load_one_api():
    with open("sikka-apis.json", "r", encoding="utf-8") as f:
        postman_data = json.load(f)
        return postman_data['item'][0]['item'][2]['item']