# USE AN OFFICIAL PYTHON RUNTIME AS A PARENT IMAGE
FROM python:3.12.3

# SET THE WORKING DIRECTORY IN THE CONTAINER
WORKDIR /app

# COPY THE CURRENT DIRECTORY CONTENTS INTO THE CONTAINER AT /app
COPY . /app

# INSTALL ANY NEEDED PACKAGES
RUN pip install requests

RUN pip install pyarrow

RUN pip install librosa

RUN pip install torch

RUN pip install transformers

RUN pip install fastapi

RUN pip install starlette

RUN pip install soundfile

RUN pip install streamlit

RUN pip install nltk

RUN pip install sentence_transformers

RUN pip install uvicorn

RUN pip install python-multipart

RUN pip install pydub

RUN pip install textblob

RUN pip install gunicorn


# MAKE PORT 80 AVAILABLE TO THE WORLD OUTSIDE THIS CONTAINER
EXPOSE 80

# RUN GUNICORN SERVER WITH UVICORN WORKERS FOR PRODUCTION
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "fast_api:app", "--bind", "0.0.0.0:80"]
