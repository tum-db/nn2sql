# NN2SQL: Let SQL Think for Neural Networks

## Scaling the Number of Input Tuples
    $ cd iris
    $ python3 iris.py > numpy_nn.csv
    $ ./iris_sql92_bench.sh | <path/to/umbra/build/sql>
    $ ./iris_sql92_psql_bench.sh | psql
    $ Rscript iris.r

## Image Classification
    $ cd mnist
    $ python3 mnist_bench.py
    $ ./mnist_sql92_bench.sh | <path/to/umbra/build/sql>
    $ <path/to/umbra/build/sql> < mnist_bench.sql
    $ Rscript mnist.r
