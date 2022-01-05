# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}


RUN apt-get update && apt-get -y install htop


RUN usermod -l keisuke vscode
RUN groupmod -n keisuke vscode
RUN echo "root:root" | chpasswd
RUN echo "keisuke:keisuke" | chpasswd
RUN usermod -aG sudo keisuke 
USER keisuke
WORKDIR /home/$USERNAME/
