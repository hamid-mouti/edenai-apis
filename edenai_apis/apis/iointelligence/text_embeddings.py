import os
import requests
from dotenv import load_dotenv

load_dotenv()

headers = {
    'Authorization': 'Bearer ' + os.getenv('IOINTELLIGENCE_API_KEY', ''),
    'Content-Type': 'application/json',
}

json_data = {
    'input': 'The food was delicious and the waiter...',
    'model': 'BAAI/bge-multilingual-gemma2',
    'encoding_format': 'float',
}

response = requests.post('https://api.intelligence.io.solutions/api/v1/embeddings', headers=headers, json=json_data)

# print response and other info
print("Status code:", response.status_code)
print("Response JSON:", response.json())