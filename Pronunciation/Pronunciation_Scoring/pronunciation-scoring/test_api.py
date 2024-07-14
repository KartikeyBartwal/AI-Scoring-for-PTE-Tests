import requests

# URL of the Hugging Face Space endpoint
url = "https://huggingface.co/spaces/Kartikeyssj2/pronunciation-scoring/check_get"

# Sending the GET request to the endpoint
response = requests.get(url)

# Checking if the request was successful
if response.status_code == 200:
    # Printing the JSON response
    print("Success:", response.json())
else:
    # Printing the error status code and text if the request failed
    print("Failed:", response.status_code, response.text)
