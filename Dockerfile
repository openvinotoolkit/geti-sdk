FROM python:3.8

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

WORKDIR /app

COPY requirements/requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD /bin/bash

FROM docker.io/ubuntu:22.04 AS development

# Install Base Dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        curl \
        git \
        build-essential \
        libedit-dev \
        libreadline-dev \
        libssl-dev \
        libbz2-dev \
        zlib1g-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        libglib2.0-0 \
        libgl1 \
        locales \
        python3 \
        python3-dev \
        python3-venv \
        python-is-python3

# Set Locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install Pyenv
RUN curl https://pyenv.run | bash
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Python 3.10.12
RUN pyenv install 3.10.12

# Install Pipenv
RUN curl https://raw.githubusercontent.com/pypa/pipenv/master/get-pipenv.py | python

# Command
CMD bash
