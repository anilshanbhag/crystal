select d_year,c_nation, COUNT(*) as profit
from lineorder,supplier,customer,part,ddate
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_region = 1
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,c_nation;
