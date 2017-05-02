FROM docker-registry.nowth.is/caffe:gpu

RUN apt update && apt-get install -y ffmpeg 

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -U pip
RUN pip3 install -r ./requirements.txt

WORKDIR /workspace
