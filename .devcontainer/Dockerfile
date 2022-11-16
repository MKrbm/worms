# https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

FROM ubuntu:22.04 as base_ubuntu


ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /home/project
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
RUN apt-get update -y && apt-get -y install htop build-essential cmake clang libssl-dev vim openssh-server cmake-curses-gui git

# n* install boost for c++
RUN apt-get -y install libboost-dev

# n* git setup
RUN git config --global --add safe.directory /home/project
RUN git config --global core.autocrlf input

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



