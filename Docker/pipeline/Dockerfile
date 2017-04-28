FROM victorhcm/opencv

WORKDIR /app

RUN apt-get update && apt-get install -y libtcmalloc-minimal4 && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:mc3man/trusty-media; sudo apt-get update; sudo apt-get install --only-upgrade ffmpeg

COPY requirements.txt /app/requirements.txt
RUN pip install -U pip
RUN pip install -r ./requirements.txt

COPY . /app
