FROM kei0709/ubuntu22.04-cpp:${TARGETARCH}
ARG USERNAME=user
USER ${USERNAME}
COPY python-package.sh python-package.sh
COPY ssh-find-agent/ /home/${USERNAME}/ssh-find-agent/
# RUN cat ssh-find-agent.sh >> $HOME/.bashrc
COPY ssh-bashrc.sh ssh-bashrc.sh
RUN cat ssh-bashrc.sh >> ~/.bashrc
RUN echo "source /usr/share/bash-completion/completions/git \n" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN ./python-package.sh
USER root
RUN apt-get install -y libopenmpi-dev liblapack-dev libblas-dev libboost-dev libeigen3-dev libboost-mpi-dev libboost-serialization-dev libboost-filesystem-dev libboost-system-dev

# setting the locale to UTF-8
RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8