FROM python:3.8

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD /bin/bash
