FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .


# Use 4 worker processes to handle requests efficiently.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "fast_api:app"]
