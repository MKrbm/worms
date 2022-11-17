FROM ubuntu:22.04 as base
ARG TARGETARCH
RUN echo "I'm building for $TARGETARCH"
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && apt-get -y install git vim software-properties-common htop cmake openssh-server
COPY terminal-colors-branch.sh /tmp/
RUN cat /tmp/terminal-colors-branch.sh >> ~/.bashrc
# # # install some utilities
RUN apt-get update -y && apt-get -y install software-properties-common &&\
  add-apt-repository ppa:deadsnakes/ppa
  

# n* install python 3.9
ARG PY_VERSION="3.9"
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python${PY_VERSION}-distutils python${PY_VERSION}-dev python${PY_VERSION} 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PY_VERSION} 1 && update-alternatives --install /usr/bin/python python /usr/bin/python${PY_VERSION} 1 && \
    apt-get -y install python3-pip

# n* install c++
RUN apt-get update -y && apt-get -y install htop build-essential cmake clang libssl-dev vim openssh-server cmake-curses-gui git curl gdb

# n* install boost for c++
RUN apt-get -y install libboost-dev

# n* git setup
RUN git config --global --add safe.directory /home/project
RUN git config --global core.autocrlf input

ARG CONDA_VER=latest


# x86
FROM base as base-amd64
ARG OS_TYPE=x86_64
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda


# aarch64
FROM base as base-arm64
ARG OS_TYPE=aarch64
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

FROM base-${TARGETARCH} as target
WORKDIR /home/project
