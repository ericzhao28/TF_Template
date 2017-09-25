FROM ubuntu:16.04

# Install necessary packages
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y
RUN apt install ca-certificates gcc make libc-dev g++ bzip2 libssl-dev build-essential python3 pip3 -y

# Hop into /service
WORKDIR /service

# Install Python requirements
COPY ./template/requirements.txt ./requirements.txt
RUN export C_INCLUDE_PATH=/usr/include
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

# Hop into /
WORKDIR /

