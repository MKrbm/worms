# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

RUN usermod -l keisuke vscode
RUN groupmod -n keisuke vscode
RUN echo "root:root" | chpasswd
RUN echo "keisuke:keisuke" | chpasswd
RUN usermod -aG sudo keisuke 
# RUN mv /home/vscode /home/keisuke
RUN usermod -d /home/keisuke -m keisuke


RUN apt-get update && apt-get -y install htop cmake
RUN apt-get -y install openssh-server
RUN apt-get -y install gfortran
RUN apt-get install cmake-curses-gui

ENV PATH="/home/keisuke/miniconda3/bin:$PATH"
ARG PATH="/home/keisuke/miniconda3/bin:$PATH"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*


WORKDIR /home/keisuke
USER keisuke

RUN wget -P /home/keisuke \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /home/keisuke/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b\
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PATH=$CONDA_DIR/envs:$PATH
RUN conda init

ADD environment.yml environment.yml
RUN conda env create -f environment.yml
RUN rm -rf environment.yml




# Put conda in path so we can use conda activate





COPY ssh-agent-manage.sh /usr/local/bin/ssh-agent-manage.sh
RUN mkdir -p ~/.config && touch ~/.config/ssh-agent.pid
RUN echo "$(cat /usr/local/bin/ssh-agent-manage.sh)" >> ~/.bashrc
WORKDIR /home/keisuke/worms

RUN chown -R keisuke:keisuke .