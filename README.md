Here are the steps to use the API for the model depoyed on EC2 instance:

A) If you ever stop the EC2 instance and the start it again, you will have to reconfigure the NGINX file. This is because the IPv4 address of the machine changes. Here is how you update the IP address:


Open the Nginx configuration file: 

```bash
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
```
Check your public IPv4 address and edit its code in server_name, replacing the old IPv4 address.

```nginx
server {
    listen 80;  # Listens for incoming traffic on port 80 (HTTP)
    server_name 44.193.226.185;  # Specifies the server name (or IP address)

    client_max_body_size 50M;  # Sets the maximum allowed size of the client request body to 50 megabytes

    location / {  # Applies to all requests to the root location
        proxy_pass http://127.0.0.1:8000;  # Forwards requests to the FastAPI application running on localhost on port 8000
    }
}
```

Save and close. Now restart the nginx server

```bash
sudo systemctl restart nginx
```

B) Next, you need to switch on the API process. To ensure the API stays awake all the time when the machine is active, I made the fastapi as a system_d process. This means you won't have to worry for it to crash anytime.

All the nginx configuration has been done. Just run the following command:

```bash
sudo systemctl start fastapi
```

You can check the status of the API with this:

```bash
sudo systemctl status fastapi
```
Note that this status command doesn't reload, so you need to re run this to see the updates.

# Configuration of the API:

## Endpoint

The API exposes a POST endpoint at `/speech_scoring/`.

## Input Parameters

- **Speech Topic**: A string that indicates the topic of the speech.
- **Audio File**: An uploaded audio file containing the speech to be evaluated.

## Process

1. The API saves the uploaded audio file temporarily and converts it to text through transcription.
2. It then analyzes the transcription for pronunciation and fluency scores.
3. The API calculates how relevant the content of the speech is to the specified topic.
4. Finally, it combines these analyses to produce a comprehensive set of scores, including pronunciation, fluency, and content relevance.


## Output

The API returns a JSON object containing the evaluation results, including scores for pronunciation, fluency, and relevance, which users can utilize for feedback or further processing.

Based on the set configuration, your API calls will be made on:

```
http://your_public_ipv4/speech_scoring/
```
