# Toolchain of build_eg2_ffmpeg.sh: Ubuntu 22.04 (glibc 2.35), so that the libraries load on Ubuntu 24.04 (glibc 2.39)
# and on newer systems. The base image is pinned by digest and the packages by an Ubuntu snapshot date.
# apt checks the snapshot's signed indexes itself; the base image has no CA certificates to check the connection with.
FROM ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11
RUN sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu/?#https://snapshot.ubuntu.com/ubuntu/20261103T000000Z/#' /etc/apt/sources.list \
    && apt-get -o Acquire::https::Verify-Peer=false update \
    && apt-get -o Acquire::https::Verify-Peer=false install -y --no-install-recommends \
        binutils gcc libc6-dev make meson nasm ninja-build pkg-config xz-utils \
    && dpkg-query -W -f '${Package} ${Version}\n' > /toolchain-packages.txt \
    && rm -rf /var/lib/apt/lists/*
