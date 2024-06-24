FROM python:3.11.7 

COPY . .

WORKDIR /

RUN pip install --no-cache-dir --upgrade -r /requirements.txt 

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "7860"]
