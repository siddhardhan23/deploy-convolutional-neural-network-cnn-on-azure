import requests
import base64
import json

url = "<endpoint-uri>"

aml_token = "<aml_token>"


# Create headers with the API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {aml_token}"
}

# Read the image
with open('test_apple_black_rot.jpeg', 'rb') as image_file:
    image_data = image_file.read()

# Convert image to base64
image_base64 = base64.b64encode(image_data).decode('utf-8')

# Construct the JSON payload
json_payload = {
    "data": image_base64
}

# Send the POST request with the JSON payload
response = requests.post(url, json=json_payload, headers=headers)

# Convert the string to a dictionary
prediction = json.loads(response.json())

# Now dict_data is a Python dictionary
print(prediction)
