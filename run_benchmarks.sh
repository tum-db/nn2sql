#!/bin/bash

export LD_LIBRARY_PATH=../umbra/lib/
UMBRASQL=../umbra/bin/sql

rm -f iris/gd_nn.csv
rm -f mnist/gd_mnist.csv

# run python baseline
(cd iris && python3 iris_bench.py > numpy_nn.csv)
(cd mnist && python3 mnist_bench.py > numpy_mnist.csv)

# run Umbra
(cd iris && ./iris_sql92_bench.sh | $UMBRASQL)
(cd mnist && $UMBRASQL < mnist_bench.sql)
(cd mnist && ./mnist_sql92_bench.sh | $UMBRASQL)

# PSQL (takes time)
echo "create or replace function sig(x float) returns float as 'select 1::float/(1+exp(-x))' LANGUAGE 'sql';" | psql
(cd iris && ./iris_sql92_psql_bench.sh | psql)

# create plots
sed -i 's/execution_time_min,/execution_time,/' mnist/gd_mnist.csv iris/gd_nn.csv
sed -i 's/name,/name,atts,limit,lr,iter,threads,/' mnist/gd_mnist.csv iris/gd_nn.csv

(cd iris && Rscript iris.r)
(cd mnist && Rscript mnist.r)
