FROM public.ecr.aws/lts/ubuntu:20.04_stable
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN apt-get update
RUN apt-get -y install linux-headers-$(uname -r)
RUN apt-get -y install build-essential git tar python3-minimal python3-pip python3-setuptools 
RUN apt-get -y install wget gnupg2

### Install nginx notebook
RUN apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN echo "deb https://apt.repos.neuron.amazonaws.com focal main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt update
RUN apt-get -y install aws-neuronx-dkms=2.* \
    aws-neuronx-tools=2.* \
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.*
RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com 
RUN pip3 install torch==1.13.1 \
    neuronx-cc==2.7.0.40 \
    transformers==4.31.0 \
    transformers-neuronx==0.5.58 \
    torch-neuronx==1.13.1.1.8.0 \
    diffusers==0.14.0 \
    accelerate>=0.20.3 \
    flask \
    gunicorn \
    pillow \
    gevent
ENV PATH=/opt/aws/neuron/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH
ADD sd21-model-wts /sd21-model-wts
COPY resources /
ENTRYPOINT ["python3", "serve.py"]
