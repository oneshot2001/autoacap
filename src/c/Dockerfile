ARG ARCH=aarch64
ARG SDK_VERSION=12.9.0
ARG UBUNTU_VERSION=24.04

FROM axisecp/acap-native-sdk:${SDK_VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION} AS builder

WORKDIR /opt/app

COPY app.c .
COPY Makefile .
COPY manifest.json .
COPY LICENSE .

RUN . /opt/axis/acapsdk/environment-setup* && \
    make

# Package as .eap
RUN . /opt/axis/acapsdk/environment-setup* && \
    acap-build . -m manifest.json
