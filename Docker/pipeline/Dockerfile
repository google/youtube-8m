FROM docker-registry.nowth.is/caffe:gpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -U pip
RUN pip install -r ./requirements.txt

COPY . /app
