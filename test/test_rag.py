import requests
import json

URL = "http://127.0.0.1:8000/rag"

payload = {
    "message": "i want to build a coding platform in nodejs for practice ",
    "user_id": "test_user_8"
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
