FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dispatcher.py .

CMD ["uvicorn", "dispatcher:app", "--host", "0.0.0.0", "--port", "8080"]