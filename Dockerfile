FROM python:3.6.10-slim

MAINTAINER Duong Tuan Linh

WORKDIR /app

COPY './requirements.txt' .

# RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
