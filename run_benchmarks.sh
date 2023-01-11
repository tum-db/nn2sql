#!/bin/bash

export LD_LIBRARY_PATH=../umbra/lib/
UMBRASQL=../umbra/bin/sql

cd iris
python3 iris_bench.py > numpy_nn.csv
./iris_sql92_bench.sh | $UMBRASQL
./iris_sql92_psql_bench.sh | psql
Rscript iris.r

cd ../mnist
python3 mnist_bench.py > numpy_mnist.csv
./mnist_sql92_bench.sh | $UMBRASQL
$UMBRASQL < mnist_bench.sql
Rscript mnist.r
