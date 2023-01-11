FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive

RUN set -x \
   && apt-get update -qq \
   && apt-get install -y --no-install-recommends \
      python3 python3-pip r-base postgresql  xz-utils\
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy
COPY . /nn2sql
RUN cd /nn2sql/mnist && tar -xzf mnist_train.csv.tgz
RUN cd /nn2sql && tar -xf umbra.tar.xz 
WORKDIR /nn2sql
CMD ["/nn2sql/run_benchmarks.sh"]

