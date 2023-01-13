# NN2SQL: Let SQL Think for Neural Networks

## clone the repository
    $ git clone https://gitlab.db.in.tum.de/MaxEmanuel/nn2sql

## Quickstart with docker
    $ docker build . nn2sql
    $ mkdir <pathforfigures>
    $ docker run -v "<pathforfigures>":/figures nn2sql

## Quickstart without docker
    $ ./prerun.sh         # extracts data + binaries
    $ ./run_benchmarks.sh # runs the benchmarks

## Reproduce also long-running SQL-92 queries in Umbra
(un)comment in run_benchmarks.sh:

    #export MNISTATTSS="20"
    export MNISTATTSS="20 200"
    #export MNISTLIMITS="200 2000"
    export MNISTLIMITS="2 20 200 2000"

## Slowstart
### Preparing
    $ sudo apt-get install python3 python3-pip r-base
    $ pip3 install numpy
    $ tar -xf umbra.tar.xz
    $ cd mnist && tar -xzf mnist_train.csv.tgz

### Scaling the Number of Input Tuples
    $ cd iris
    $ python3 iris_bench.py > numpy_nn.csv
    $ ./iris_sql92_bench.sh | ../umbra/bin/sql
    $ ./iris_sql92_psql_bench.sh | psql
    $ Rscript iris.r

### Image Classification
    $ cd mnist
    $ python3 mnist_bench.py > numpy_mnist.csv
    $ ./mnist_sql92_bench.sh | ../umbra/bin/sql
    $ ../umbra/bin/sql < mnist_bench.sql
    $ Rscript mnist.r
