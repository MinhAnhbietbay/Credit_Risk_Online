FROM python:3.12-slim
WORKDIR /app
# lightgbm/xgboost cần libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# src/, scripts/, data/, outputs/ được mount qua docker-compose (data ~3.7GB, không copy vào image)
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
