FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        tmux \
        vim \
        git \
        curl \
        wget \
        openssh-client \
        rsync \
        procps \
        coreutils \
        ca-certificates \
        docker.io && \
    rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
