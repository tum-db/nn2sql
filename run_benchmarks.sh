#!/bin/bash

export LD_LIBRARY_PATH=../umbra/lib/
UMBRASQL=../umbra/bin/sql

# run python baseline
(cd iris && python3 iris_bench.py > numpy_nn.csv)
(cd mnist && python3 mnist_bench.py > numpy_mnist.csv)

# run Umbra
(cd iris && ./iris_sql92_bench.sh | $UMBRASQL)
(cd mnist && $UMBRASQL < mnist_bench.sql)
(cd mnist && ./mnist_sql92_bench.sh | $UMBRASQL)

# PSQL (takes time)
(cd iris && ./iris_sql92_psql_bench.sh | psql)

# create plots
(cd iris && Rscript iris.r)
(cd mnist && Rscript mnist.r)
