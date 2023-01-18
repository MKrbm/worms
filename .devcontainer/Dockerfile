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