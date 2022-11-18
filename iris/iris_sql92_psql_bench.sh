#!/bin/bash

# parameters
parallel=${PARALLEL:-"8"}
lr=0.2 # learningrate
attss="20 50"
iters="10 100 1000"
limits="150 300 600 1200"
repeat=3

echo "
drop table iris;
drop table iris2;
drop table img;
drop table one_hot;
drop table w_xh;
drop table w_ho;
create table iris (sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;
\copy iris from './iris.csv' delimiter ',' HEADER CSV;

create table if not exists iris2 (id int, sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
insert into iris2 (select row_number() over (), * from iris);
create table if not exists img (i int, j int, v float);
create table if not exists one_hot(i int, j int, v int, dummy int);
insert into img (select id,1,sepal_length/10 from iris2);
insert into img (select id,2,sepal_width/10 from iris2);
insert into img (select id,3,petal_length/10 from iris2);
insert into img (select id,4,petal_width/10 from iris2);
insert into one_hot(select n.i, n.j, coalesce(i.v,0), i.v from (select id,species+1 as species,1 as v from iris2) i right outer join (select a.a as i, b.b as j from (select generate_series as a from generate_series(1,150)) a, (select generate_series as b from generate_series(1,4)) b) n on n.i=i.id and n.j=i.species order by i,j);

create table if not exists w_xh (w_id int, i int, j int, v float);
create table if not exists w_ho (w_id int, i int, j int, v float);

"
for atts in $attss; do
echo "
insert into w_xh (select $atts, i.*,j.*,random()*2-1 from generate_series(1,4) i, generate_series(1,$atts) j);
insert into w_ho (select $atts, i.*,j.*,random()*2-1 from generate_series(1,$atts) i, generate_series(1,3) j);
"
done
echo "\timing true"
for limit in $limits; do
  for iter in $iters; do
    for atts in $attss; do
echo "
with recursive w (iter,id,i,j,v) as (
  (select 0,0,i,j,v from w_xh where w_id=$atts union select 0,1,i,j,v from w_ho where w_id=$atts)
  union all
  (
  with w_now as (
     SELECT * from w
  ), a_xh(i,j,v) as (
     SELECT m.i, n.j, sig(SUM (m.v*n.v)) --1/(1+exp(-SUM (m.v*n.v)))
     FROM img AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE m.i < $limit and n.id=0 and n.iter=(select max(iter) from w_now) -- w_xh
     GROUP BY m.i, n.j
  ), a_ho(i,j,v) as (
     SELECT m.i, n.j, sig(SUM (m.v*n.v)) --1/(1+exp(-SUM (m.v*n.v))) --sig(SUM (m.v*n.v))
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
     WHERE m.i < $limit
     GROUP BY m.j, n.j
     union
     SELECT 1, m.j as i, n.j, (SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN d_ho AS n ON m.i=n.i
     GROUP BY m.j, n.j
  )
  select iter+1, w.id, w.i, w.j, w.v - $lr * d_w.v
  from w_now as w, d_w
  where iter < $iter and w.id=d_w.id and w.i=d_w.i and w.j=d_w.j
  )
)
select count(*) from w;
"
done
done
done
