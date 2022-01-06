# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}


RUN apt-get update && apt-get -y install htop cmake
RUN apt-get -y install openssh-server
RUN apt-get -y install gfortran

RUN usermod -l keisuke vscode
RUN groupmod -n keisuke vscode
RUN echo "root:root" | chpasswd
RUN echo "keisuke:keisuke" | chpasswd
RUN usermod -aG sudo keisuke 
# RUN mv /home/vscode /home/keisuke
RUN usermod -d /home/keisuke -m keisuke
USER keisuke
COPY ssh-agent-manage.sh /usr/local/bin/ssh-agent-manage.sh
RUN mkdir -p ~/.config && touch ~/.config/ssh-agent.pid
RUN echo "$(cat /usr/local/bin/ssh-agent-manage.sh)" >> ~/.bashrc
WORKDIR /home/keisuke/worms
