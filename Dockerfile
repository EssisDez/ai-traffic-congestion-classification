FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY traffic_congestion_mobilenetv2.keras .
COPY class_names.json .

EXPOSE 8080

CMD ["python", "app.py"]