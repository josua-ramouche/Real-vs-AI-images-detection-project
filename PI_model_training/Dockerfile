FROM python:3.11-slim

RUN apt-get update -y && apt-get install -y \
    nano \
    net-tools \
    iputils-ping \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8090

CMD ["uvicorn", "api_server:app", "--reload", "--host", "172.19.0.11", "--port", "8090"]