create table lineorder (
   lo_orderkey integer not null,
   lo_linenumber integer not null,
   lo_custkey integer not null,
   lo_partkey integer not null,
   lo_suppkey integer not null,
   lo_orderdate integer not null,
   lo_orderpriority char(15) not null,
   lo_shippriority char(1) not null,
   lo_quantity integer not null,
   lo_extendedprice float not null,
   lo_ordtotalprice float not null,
   lo_discount float not null,
   lo_revenue float not null,
   lo_supplycost float not null,
   lo_tax integer not null,
   lo_commitdate integer not null,
   lo_shopmode char(10) not null,
   primary key (lo_orderkey,lo_linenumber)
);

create table part (
   p_partkey integer not null,
   p_name varchar(22) not null,
   p_mfgr char(6) not null,
   p_category char(7) not null,
   p_brand1 char(9) not null,
   p_color varchar(11) not null,
   p_type varchar(25) not null,
   p_size integer not null,
   p_container char(10) not null,
   primary key (p_partkey)
);

create table supplier (
   s_suppkey integer not null,
   s_name char(25) not null,
   s_address varchar(25) not null,
   s_city char(10) not null,
   s_nation char(15) not null,
   s_region char(12) not null,
   s_phone char(15) not null,
   primary key (s_suppkey)
);

create table customer (
   c_custkey integer not null,
   c_name varchar(25) not null,
   c_address varchar(25) not null,
   c_city char(10) not null,
   c_nation char(15) not null,
   c_region char(12) not null,
   c_phone char(15) not null,
   c_mktsegment char(10) not null,
   primary key (c_custkey)
);

create table ddate (
   d_datekey integer not null,
   d_date char(18) not null,
   d_dayofweek char(9) not null,
   d_month char(9) not null,
   d_year integer not null,
   d_yearmonthnum integer not null,
   d_yearmonth char(7) not null,
   d_daynuminweek integer not null,
   d_daynuminmonth integer not null,
   d_daynuminyear integer not null,
   d_monthnuminyear integer not null,
   d_weeknuminyear integer not null,
   d_sellingseasin varchar(12) not null,
   d_lastdayinweekfl integer not null,
   d_lastdayinmonthfl integer not null,
   d_holidayfl integer not null,
   d_weekdayfl integer not null,
   primary key (d_datekey)
);
