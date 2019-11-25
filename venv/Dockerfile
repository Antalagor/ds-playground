FROM ubuntu

RUN apt-get -y update && \
    apt-get -y install \
    git \
    python3-pip \
    libopenmpi-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

ENV VENV /root/venv

RUN pip3 install virtualenv && \
    virtualenv $VENV && \
    . $VENV/bin/activate

ENV PATH=$VENV/bin:$PATH

ADD requirements.txt .

RUN pip install -r requirements.txt

CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root $MOUNT

EXPOSE 8888
