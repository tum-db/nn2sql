import duckdb
import numpy as np
from datetime import datetime
from duckdb.typing import *

rep=1
loadlimit=6000
sizes=[2000,200,20,2]
attss=[20]
learningrate=0.2

createschema = '''
drop table if exists img;
drop table if exists one_hot;
drop table if exists mnist;
drop table if exists mnist2;
create table if not exists img (i int, j int, v float);
create table if not exists one_hot(i int, j int, v int, dummy int);'''

loadmnist = 'create table mnist (label float'
loadmnist2 = 'create table mnist2 (id int, label float'
loadmnistrel = '''
copy mnist from './mnist_train.csv' delimiter ',' HEADER CSV;
insert into mnist2 (select row_number() over (), * from mnist limit {});
insert into one_hot(select n.i, n.j, coalesce(i.v,0), i.v from (select id,label+1 as species,1 as v from mnist2) i right outer join (select a.a as i, b.b as j from (select generate_series as a from generate_series(1,{})) a, (select generate_series as b from generate_series(1,10)) b) n on n.i=i.id and n.j=i.species order by i,j);
'''
for i in range(1,785):
	loadmnist += ', pixel{} float'.format(i)
	loadmnist2 += ', pixel{} float'.format(i)
	loadmnistrel += 'insert into img (select id,{},pixel{}/255 from mnist2); '.format(i,i)
loadmnist += ');'
loadmnist2 += ');'

weights ='''
drop table if exists w_xh;
drop table if exists w_ho;
create table if not exists w_xh (i int, j int, v float);
create table if not exists w_ho (i int, j int, v float);
insert into w_xh (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);
insert into w_ho (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);'''

train ='''with recursive w (iter,id,i,j,v) as (
  (select 0,0,* from w_xh union select 0,1,* from w_ho)
  union all
  (
  with w_now as (
     SELECT * from w
  ), a_xh(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v)))
     FROM img AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE m.i < {} and n.id=0 and n.iter=(select max(iter) from w_now) -- w_xh
     GROUP BY m.i, n.j
  ), a_ho(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) --sig(SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.j
  ), l_ho(i,j,v) as (
     select m.i, m.j, 2*(m.v-n.v)
     from a_ho AS m INNER JOIN one_hot AS n ON m.i=n.i AND m.j=n.j
  ), d_ho(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_ho AS m INNER JOIN a_ho AS n ON m.i=n.i AND m.j=n.j
  ), l_xh(i,j,v) as (
     SELECT m.i, n.i as j, (SUM (m.v*n.v)) -- transpose
     FROM d_ho AS m INNER JOIN w_now AS n ON m.j=n.j
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.i
  ), d_xh(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_xh AS m INNER JOIN a_xh AS n ON m.i=n.i AND m.j=n.j
  ), d_w(id,i,j,v) as (
     SELECT 0, m.j as i, n.j, (SUM (m.v*n.v))
     FROM img AS m INNER JOIN d_xh AS n ON m.i=n.i
     WHERE m.i < {}
     GROUP BY m.j, n.j
     union
     SELECT 1, m.j as i, n.j, (SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN d_ho AS n ON m.i=n.i
     GROUP BY m.j, n.j
  )
  select iter+1, w.id, w.i, w.j, w.v - {} * d_w.v
  from w_now as w, d_w
  where iter < {} and w.id=d_w.id and w.i=d_w.i and w.j=d_w.j
  )
)'''
justprint='''SELECT DISTINCT iter FROM w;'''
label='''SELECT iter, count(*)::float/(select count(distinct i) from one_hot) as precision
FROM (
   SELECT *, rank() over (partition by m.i,iter order by v desc) as rank
   FROM (
      SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) as v, m.iter
      FROM (
         SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) as v, iter
         FROM img AS m INNER JOIN w AS n ON m.j=n.i
         WHERE n.id=0 -- and n.iter=(select max(iter) from w)
         GROUP BY m.i, n.j, iter ) AS m INNER JOIN w AS n ON m.j=n.i
      WHERE n.id=1 and n.iter=m.iter
      GROUP BY m.i, n.j, m.iter
   ) m ) pred,
   (SELECT *, rank() over (partition by m.i order by v desc) as rank FROM one_hot m) test
WHERE pred.i=test.i and pred.rank = 1 and test.rank=1
GROUP BY iter, pred.j=test.j
HAVING (pred.j=test.j)=true
ORDER BY iter
'''
labelmax='SELECT max(precision) FROM (' +  label + ')'


def benchmark(atts,limit,iterations,learning_rate):
	duckdb.sql(createschema)
	duckdb.sql(loadmnist)
	duckdb.sql(loadmnist2)
	duckdb.sql(loadmnistrel.format(limit,limit))
	duckdb.sql(weights.format(784,atts,atts,10))
	loadtime = datetime.now()
	start = datetime.now()
	for i in range(rep):
		result = duckdb.sql(train.format(limit,limit,learning_rate,iterations) + labelmax).fetchall()
	time=(datetime.now() - start).total_seconds()/rep
	#name,atts,limit,lr,iter,execution_time,accuracy
	print("DuckDB-SQL-92,{},{},{},{},{},{}".format(atts,limit,learning_rate,iterations,time,result[0][0]))

print("name,atts,limit,lr,iter,execution_time,accuracy")
for atts in attss:
	for size in sizes:
		iterations=int(loadlimit/size)
		benchmark(atts,size,iterations,learningrate)
