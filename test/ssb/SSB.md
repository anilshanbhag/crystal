Star Schema Benchmark Queries
=============================


Queries
-------

q11

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder,date
where lo_orderdate = d_datekey
and d_year = 1993 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;

q11.m

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19930101 and lo_orderdate <= 19940101 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;

q12

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder,date
where lo_orderdate = d_datekey
and d_yearmonthnum = 199401
and lo_discount>=4
and lo_discount<=6
and lo_quantity>=26
and lo_quantity<=35;

q12.m

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19940101 and lo_orderdate <= 19940131 
and lo_discount>=4 and lo_discount<=6
and lo_quantity>=26
and lo_quantity<=35;

q13

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder,date
where lo_orderdate = d_datekey
and d_weeknuminyear = 6
and d_year = 1994
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;

q13.m

select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19940204
and lo_orderdate <= 19940210
and lo_discount>=5
and lo_discount<=7
and lo_quantity>=26
and lo_quantity<=35;

q21

select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,date
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
group by d_year,p_brand1
order by d_year,p_brand1;

q21.m

select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 1
and s_region = 1
group by d_year,p_brand1
order by d_year,p_brand1;

q22

select sum(lo_revenue),d_year,p_brand1
from lineorder, part, supplier,date
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 >= 'MFGR#2221'
and p_brand1 <= 'MFGR#2228'
and s_region = 'ASIA'
group by d_year,p_brand1
order by d_year,p_brand1;

q22.m

select sum(lo_revenue),d_year,p_brand1
from lineorder, part, supplier,ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 >= 260
and p_brand1 <= 267
and s_region = 2
group by d_year,p_brand1
order by d_year,p_brand1;

q23

select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,date
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 = 'MFGR#2239'
and s_region = 'EUROPE'
group by d_year,p_brand1
order by d_year,p_brand1;

q23.m

select sum(lo_revenue),d_year,p_brand1
from lineorder,part,supplier,ddate
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 = 278
and s_region = 3
group by d_year,p_brand1
order by d_year,p_brand1;

Dictionary Encoding
America => 1
Asia => 2
Europe => 3

q31 [Aggregates greater than int]

select c_nation,s_nation,d_year,sum(lo_revenue) as revenue
from lineorder,customer, supplier,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 'ASIA'
and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
group by c_nation,s_nation,d_year
order by d_year asc,revenue desc;

q31.m

select c_nation,s_nation,d_year,sum(lo_revenue) as revenue
from lineorder,customer, supplier,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 2
and s_region = 2
and d_year >= 1992 and d_year <= 1997
group by c_nation,s_nation,d_year
order by d_year asc,revenue desc;

q32

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

q32.m

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 24
and s_nation = 24
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

q33

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and (c_city = 'UNITED KI1' or c_city = 'UNITED KI5')
and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5')
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

q33.m

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and (c_city = 231 or c_city = 235)
and (s_city = 231 or s_city = 235)
and d_year >=1992 and d_year <= 1997
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

q34

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and (c_city = 'UNITED KI1' or c_city = 'UNITED KI5')
and (s_city = 'UNITED KI1' or s_city = 'UNITED KI5')
and d_yearmonth = 'Dec1997'
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

q34.m

select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and (c_city = 231 or c_city = 235)
and (s_city = 231 or s_city = 235)
and d_yearmonthnum = 199712
group by c_city,s_city,d_year
order by d_year asc,revenue desc;

ASIA => 2
UNITED STATES => 24
UNITED KI1 => 231
UNITED KI5 => 235

q41 [Aggregates greater than int]

select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit
from lineorder,supplier,customer,part, date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year,c_nation
order by d_year,c_nation;

q41.m

select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit
from lineorder,supplier,customer,part,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_region = 1
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,c_nation
order by d_year,c_nation;

q42

select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit
from lineorder,customer,supplier,part,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and (d_year = 1997 or d_year = 1998)
and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year,s_nation, p_category
order by d_year,s_nation, p_category;

q42.m

select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit
from lineorder,customer,supplier,part,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_region = 1
and (d_year = 1997 or d_year = 1998)
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,s_nation, p_category
order by d_year,s_nation, p_category;

q43

select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit
from lineorder,supplier,customer,part,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_nation = 'UNITED STATES'
and (d_year = 1997 or d_year = 1998)
and p_category = 'MFGR#14'
group by d_year,s_city,p_brand1
order by d_year,s_city,p_brand1;

q43.m

select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit
from lineorder,supplier,customer,part,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_nation = 24
and (d_year = 1997 or d_year = 1998)
and p_category = 3
group by d_year,s_city,p_brand1
order by d_year,s_city,p_brand1;

AMERICA => 1
MFGR#1 => 1
MFGR#2 => 2
UNITED STATES => 24

Data Generation
----------------

SF 1:
./gpuDBLoaderM --lineorder ../../test/dbgen/lineorder.tbl --ddate ../../test/dbgen/date.tbl --customer ../../test/dbgen/customer.tbl.p --supplier ../../test/dbgen/supplier.tbl.p --part ../../test/dbgen/part.tbl.p --datadir ../../dataM1/

SF 10:
./gpuDBLoaderM --customer ../../data-raw10/customer.tbl.p --supplier ../../data-raw10/supplier.tbl.p --part ../../data-raw10/part.tbl.p --datadir ../../dataM10/

SF 20:
./gpuDBLoaderM --lineorder ../../data-raw20/lineorder.tbl --ddate ../../data-raw20/date.tbl --customer ../../data-raw20/customer.tbl.p --supplier ../../data-raw20/supplier.tbl.p --part ../../data-raw20/part.tbl.p --datadir ../../dataM20/


python convert.py

Inefficiencies
-------------

* Hash function (eg: for q23)

Hyper
---

./bin/driver /big_fast_drive/anil/dbops/test/ssb/schema.sql /big_fast_drive/anil/dbops/test/ssb/load.sql --store ssb_transformed.dump
