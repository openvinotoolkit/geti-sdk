FROM python:3.8

RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv

RUN useradd --create-home --shell /bin/bash --no-log-init getisdk
USER getisdk

RUN pip install opencv-python

WORKDIR /home/getisdk

COPY requirements/requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD /bin/bash
