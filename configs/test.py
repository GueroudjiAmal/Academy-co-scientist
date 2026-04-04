"""
Argo Gateway API Sample Script
- Chat endpoint using the messages object
"""
import requests
import json

# API endpoint to POST
url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

# Data to be sent as a POST in JSON format
data = {
    "user": "amal",
    "model": "gpt54",
    "messages": [
        {"role": "system",
          "content": "You are a large language model with the name Argo"},
        {"role": "user",
          "content": "What is your name?"},
        {"role": "assistant",
          "content": "My name is Argo."},
        {"role": "user",
          "content": "I want you to generate a hypothesis on how to use vectordbs to improve knowledge graph construction."}
    ],
    "stop": [],
    "temperature": 0.1,
    "top_p": 0.9,
}

# Convert the dict to JSON
payload = json.dumps(data)

# Add a header stating that the content type is JSON
headers = {"Content-Type": "application/json"}

# Send POST request
response = requests.post(url, data=payload, headers=headers)

# Receive the response data
print("Status Code:", response.status_code)
print("LLM response: ", response.json()["response"])
