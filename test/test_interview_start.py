import requests

payload = {
    "user_id": "himansh",
    "role": "Backend Developer",
    "experience": 1,
    "job_description": "Node.js, Express, REST APIs, MongoDB",
    "difficulty": "medium"
}

r = requests.post("http://localhost:8000/interview_text/start", json=payload)
print(r.json())
