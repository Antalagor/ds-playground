FROM python:3.7

RUN apt-get -y update && \
    apt-get -y install \
    git \
    python3-pip \
    libopenmpi-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

ADD requirements.txt .
ADD jupyter_notebook_config.py .

RUN pip3 install -r requirements.txt

CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root --config=jupyter_notebook_config.py $PYTHONPATH

EXPOSE 8888
