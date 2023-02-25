FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y gcc && \
#    python3 -m venv venv && source venv/bin/activate \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "ind:app", "-b", "0.0.0.0:8000"]
