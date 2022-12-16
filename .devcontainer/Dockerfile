FROM kei0709/ubuntu22.04-cpp:${TARGETARCH}
ARG USERNAME=user
USER ${USERNAME}
COPY python-package.sh python-package.sh
COPY ssh-find-agent/ /home/${USERNAME}/ssh-find-agent/
# RUN cat ssh-find-agent.sh >> $HOME/.bashrc
RUN echo ". ~/ssh-find-agent/ssh-find-agent.sh # for bash \n \
    ssh_find_agent -a \n \
    if [ -z "$SSH_AUTH_SOCK" ] \n \
    then \n" >> $HOME/.bashrc
RUN echo 'eval $(ssh-agent) > /dev/null \n' >> $HOME/.bashrc
RUN echo "ssh-add -l >/dev/null || alias ssh='ssh-add -l >/dev/null || ssh-add && unalias ssh; ssh' \n \
    fi \n " \
    >> $HOME/.bashrc

SHELL ["/bin/bash", "-c"]
RUN ./python-package.sh