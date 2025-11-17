import requests
import json

URL = "http://127.0.0.1:8000/rag"

payload = {
    "message": "tell me about all the courses you are offering",
    "user_id": "test_user_terminal"
}

headers = {
    "Content-Type": "application/json"
}

print("👉 Sending request to /rag ...\n")

try:
    response = requests.post(URL, headers=headers, data=json.dumps(payload))
    
    print("🔹 Status Code:", response.status_code)
    print("🔹 Response JSON:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("❌ Error while connecting to the microservice:")
    print(str(e))
