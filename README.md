This is the deployment branch that is the end product after all the experimentations, model creations and tuning. Here are the steps to use this:

Change to the current directory 

!pip install -r requirements.txt

python3 -m uvicorn fast_api:app 


Here is the API structure and its further details:

It has been deployed on a EC2 Machine Instance ( Ubuntu debian ). Every single time you stop and restart the instance, you will need to launch the fast_api again. Here is 
how you do it on the EC2 instance:

