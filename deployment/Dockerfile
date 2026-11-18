FROM python:3.9-slim

WORKDIR .

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY model /model
COPY model_data /model_data

EXPOSE 5000

CMD ["python3", "app.py"]
