FROM python:3.8-bullseye@sha256:5277b878d90029446c0124f08064ff3eba05a4d5ce182ee4b17a52a32d83d392

RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv libgl1

RUN useradd --create-home --user-group --shell /bin/bash --no-log-init getisdk
USER getisdk

WORKDIR /home/getisdk
ENV HOME /home/getisdk
ENV PATH="$PATH:$HOME/.local/bin"

RUN python -m pip install --upgrade pip
RUN python -m pip install opencv-python

COPY --chown=getisdk:getisdk . $HOME/geti-sdk
RUN python -m pip install $HOME/geti-sdk

ENTRYPOINT /bin/bash
