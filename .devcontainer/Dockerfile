FROM kei0709/ubuntu22.04-cpp:${TARGETARCH}
USER user
COPY python-package.sh python-package.sh
SHELL ["/bin/bash", "-c"]
RUN ./python-package.sh
# FROM --platform=linux/arm64 ubuntu:latest
