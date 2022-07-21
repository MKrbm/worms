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
  

# n* install python 3.9
RUN apt-get -y install python${PY_VERSION}-distutils python${PY_VERSION}-dev python${PY_VERSION} 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PY_VERSION} 1 && update-alternatives --install /usr/bin/python python /usr/bin/python${PY_VERSION} 1 && \
    apt-get -y install python3-pip

# ADD requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt
# RUN rm -rf environment.yml
# n* installed python

RUN apt-get update -y && apt-get -y install htop cmake openssh-server cmake-curses-gui

# n* install conda
# ENV PATH="/home/$USERNAME/miniconda3/bin:$PATH"
# RUN apt-get update && \
#     apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# WORKDIR /home/$USERNAME
# USER $USERNAME

# RUN dpkgArch="$(dpkg --print-architecture)";\
# 	case "$dpkgArch" in \
# 		arm64) export CONDA='Miniconda3-latest-Linux-aarch64.sh' ;; \
# 		*) export CONDA='Miniconda3-latest-Linux-x86_64.sh' ;; \
# 	esac; \
#     wget -P /home/${USERNAME} \
#     https://repo.anaconda.com/miniconda/$CONDA \
#     && mkdir /home/$USERNAME/.conda \
#     && bash $CONDA -b\
#     && rm -f $CONDA
# ENV PATH=$CONDA_DIR/bin:$PATH
# ENV PATH=$CONDA_DIR/envs:$PATH
# RUN conda init

# install via environment.yml
# # ADD environment.yml environment.yml
# # RUN conda env create -f environment.yml
# # RUN rm -rf environment.yml
# n* installed conda




USER $USERNAME
RUN git config --global --add safe.directory /home/user/project
WORKDIR /home/${USERNAME}/project