import os
import requests
from dotenv import load_dotenv

load_dotenv()

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + os.getenv('IOINTELLIGENCE_API_KEY', ''),
}

print("IO intelligence key: ", os.getenv('IOINTELLIGENCE_API_KEY', ''))

selected_model = "mistralai/Ministral-8B-Instruct-2410"

json_data = {
    'model': selected_model,
    'messages': [
        {
            'role': 'system',
            'content': 'You are a helpful assistant.',
        },
        {
            'role': 'user',
            'content': 'bruh!',
        },
    ],
}

response = requests.post('https://api.intelligence.io.solutions/api/v1/chat/completions', headers=headers, json=json_data)

# print response and other info
print("Status code:", response.status_code)
print("Response JSON:", response.json())