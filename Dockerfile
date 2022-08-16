# This Dockerfile creates a container for running the algorithms, tests, and benchmarks in the open-source repository
# for Quantum Computing For Programmers. In order to ensure that everything is ready for running the algorithms and
# benchmarks, the library is built and the tests are run by default.
#
# Copyright 2022, Abdolhamid Pourghazi <pourgh01@ads.uni-passau.de>
# SPDX-License-Identifier: Apache-2.0

FROM debian:11

LABEL maintainer="Abdolhamid Pourghazi <pourgh01@ads.uni-passau.de>"
LABEL maintainer="Stefan Klessinger <stefan.klessinger@uni-passau.de>"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		git \
		libpython3-dev \
		python3 \
		python3-pip \
		wget


RUN python3 -m pip install absl-py numpy scipy

# Add user
RUN useradd -m -G sudo -s /bin/bash repro && echo "repro:repro" | chpasswd
USER repro

# Install bazelisk
RUN mkdir -p /home/repro/bin/
RUN wget -O /home/repro/bin/bazel 	https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
RUN chmod +x /home/repro/bin/bazel

ENV PATH="/home/repro/bin/:${PATH}"

# Get qcc source
RUN mkdir -p /home/repro/sources/
WORKDIR /home/repro/sources/
RUN git clone https://github.com/qcc4cp/qcc.git

# Update WORKSPACE
WORKDIR /home/repro/sources/qcc
RUN sed -i 's/python3.7/python3.9/g' WORKSPACE

# Build qcc
WORKDIR /home/repro/sources/qcc/src/lib
RUN bazel build all

ENV PYTHONPATH=$PYTHONPATH:/home/repro/sources/qcc/bazel-bin/src/lib

# Run tests
RUN bazel test ...
RUN bazel run circuit_test

WORKDIR /home/repro/sources/qcc/src/libq
RUN bazel test ...

# Set initial directory
WORKDIR /home/repro/sources/qcc/src