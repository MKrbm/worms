# https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

ARG VARIANT="ubuntu-22.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}
ARG USERNAME=user
ARG GROUPNAME=user
ARG PY_VERSION="3.9"
# ARG UID=1001
# ARG GID=1001
RUN usermod -l user vscode
RUN groupmod -n user vscode
RUN echo "root:root" | chpasswd
RUN echo "$USERNAME:$USERNAME" | chpasswd
RUN usermod -d /home/$USERNAME -m $USERNAME
RUN usermod -aG sudo user 
WORKDIR /home/${USERNAME}/project
RUN git config --global --add safe.directory /home/${USERNAME}/project

# # # install some utilities
RUN apt-get update -y && apt-get -y install software-properties-common &&\
  add-apt-repository ppa:deadsnakes/ppa
  

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