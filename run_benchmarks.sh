#!/bin/bash

export LD_LIBRARY_PATH=../umbra/lib/
UMBRASQL=../umbra/bin/sql
export MNISTATTSS="20"
#export MNISTATTSS="20 200"
export MNISTLIMITS="200 2000"
#export MNISTLIMITS="2 20 200 2000"

rm -f iris/gd_nn.csv
rm -f mnist/gd_mnist.csv

# run python baseline
(cd iris && python3 iris_bench.py > numpy_nn.csv)
(cd mnist && python3 mnist_bench.py > numpy_mnist.csv)

# run Umbra
(cd iris && ./iris_sql92_bench.sh | $UMBRASQL)
(cd mnist && $UMBRASQL < mnist_bench.sql)
(cd mnist && ./mnist_sql92_bench.sh | $UMBRASQL)

# run DuckDB baseline
(cd iris && python3 duckdb_iris.py > duckdb_iris.csv)
(cd mnist && python3 duckdb_mnist.py > duckdb_mnist.csv)

# PSQL (takes time)
echo "create or replace function sig(x float) returns float as 'select 1::float/(1+exp(-x))' LANGUAGE 'sql';" | psql
(cd iris && ./iris_sql92_psql_bench.sh | psql | grep Time > psql_nn.csv)

# create plots
sed -i 's/execution_time_min,/execution_time,/' mnist/gd_mnist.csv iris/gd_nn.csv
sed -i 's/name,/name,atts,limit,lr,iter,threads,/' mnist/gd_mnist.csv iris/gd_nn.csv

(cd iris && Rscript iris.r && cp iris.pdf /figures)
(cd mnist && Rscript mnist.r && cp mnist.pdf /figures)
