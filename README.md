# NN2SQL: Let SQL Think for Neural Networks

## Preparing
    $ sudo apt-get install python3 python3-pip r-base
    $ pip3 install numpy
    $ tar -xf umbra.tar.xz
    $ cd mnist && tar -xzf mnist_train.csv.tgz

## Scaling the Number of Input Tuples
    $ cd iris
    $ python3 iris_bench.py > numpy_nn.csv
    $ ./iris_sql92_bench.sh | <path/to/umbra/build/sql>
    $ ./iris_sql92_psql_bench.sh | psql
    $ Rscript iris.r

## Image Classification
    $ cd mnist
    $ python3 mnist_bench.py > numpy_mnist.csv
    $ ./mnist_sql92_bench.sh | <path/to/umbra/build/sql>
    $ <path/to/umbra/build/sql> < mnist_bench.sql
    $ Rscript mnist.r
