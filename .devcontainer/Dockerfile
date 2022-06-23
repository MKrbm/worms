# https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

ARG VARIANT="ubuntu-22.04"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}
ARG USERNAME=user
ARG GROUPNAME=user
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
RUN apt-get update && apt-get -y install htop cmake
RUN apt-get -y install openssh-server
RUN apt-get install cmake-curses-gui

# #install fortran
RUN apt-get -y install gfortran

ENV PATH="/home/$USERNAME/miniconda3/bin:$PATH"
RUN apt-get update && \
    apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /home/$USERNAME
USER $USERNAME

RUN dpkgArch="$(dpkg --print-architecture)";\
	case "$dpkgArch" in \
		arm64) export CONDA='Miniconda3-latest-Linux-aarch64.sh' ;; \
		*) export CONDA='Miniconda3-latest-Linux-x86_64.sh' ;; \
	esac; \
    wget -P /home/${USERNAME} \
    https://repo.anaconda.com/miniconda/$CONDA \
    && mkdir /home/$USERNAME/.conda \
    && bash $CONDA -b\
    && rm -f $CONDA
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PATH=$CONDA_DIR/envs:$PATH
RUN conda init

RUN git config --global --add safe.directory /home/user/project

# install via environment.yml
# # ADD environment.yml environment.yml
# # RUN conda env create -f environment.yml
# # RUN rm -rf environment.yml

USER $USERNAME
WORKDIR /home/${USERNAME}/project