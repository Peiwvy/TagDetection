ARG ROS_DISTRO=melodic
FROM ros:$ROS_DISTRO 
# Change repository to SJTU mirror
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirror.sjtu.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's/http:\/\/packages.ros.org/http:\/\/mirror.sjtu.edu.cn/g' /etc/apt/sources.list.d/ros1-latest.list
# install basic tools & requirements
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y software-properties-common apt-utils \
    bash-completion openssh-client wget curl vim tmux git sudo \
    libgtest-dev libgoogle-glog-dev libeigen3-dev libyaml-cpp-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# install gcc
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-9 g++-9 gdb build-essential ninja-build \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# install cmake
RUN wget https://apt.kitware.com/kitware-archive.sh \
    && bash kitware-archive.sh \
    && apt-get install -y cmake \
    && rm kitware-archive.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# install llvm
RUN wget https://apt.llvm.org/llvm.sh \
    && bash llvm.sh 15 \
    && rm llvm.sh \
    && apt-get autoremove -y clang-15 lldb-15 lld-15 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 
ENV PATH="/usr/lib/llvm-15/bin:${PATH}"
# build gtest
RUN cd /usr/src/gtest \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build \
    && rm -rf build
# build fmt-9.1.0
RUN wget https://github.com/fmtlib/fmt/archive/refs/tags/9.1.0.tar.gz \
    && tar -zxvf 9.1.0.tar.gz \
    && rm 9.1.0.tar.gz \
    && cd fmt-9.1.0 \
    && cmake -B build -DFMT_TEST=OFF \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build \
    && cd .. \
    && rm -rf fmt-9.1.0
# create user
ARG USERNAME=cyber
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
# setup ros enviroments
RUN echo source /opt/ros/$ROS_DISTRO/setup.bash >> /home/$USERNAME/.bashrc
