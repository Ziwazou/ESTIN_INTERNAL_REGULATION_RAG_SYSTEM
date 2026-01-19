import requests

response = requests.post(
    "http://localhost:8000/api/v1/ask",
    json={"question": "Quelles sont les sanctions du 3ème degré?"}
)
print(response.json()["answer"])