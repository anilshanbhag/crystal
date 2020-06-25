Crystal GPU Library
=================

The Crystal library implements a collection of block-wide device functions that can be used to implement high performance implementations of SQL queries on GPUs.

The package contains:

* Crystal: `crystal/` contains the block-wide device functions
* Implementations: `src/` contains SQL query operator implementations and implementations of 13 queries of the Star Schema Benchmark

For full details of the Crystal, see our [paper](http://anilshanbhag.in/static/papers/crystal_sigmod20.pdf)

```
@inproceedings{shanbhag2020crystal,
  author = {Shanbhag, Anil and Madden, Samuel and Yu, Xiangyao},
  title = {A Study of the Fundamental Performance Characteristics of GPUs and CPUs for Database Analytics},
  year = {2020},
  url = {https://doi.org/10.1145/3318464.3380595},
  doi = {10.1145/3318464.3380595},
  booktitle = {Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data},
  pages = {1617–1632},
  numpages = {16},
  location = {Portland, OR, USA},
  series = {SIGMOD ’20}
}
```

Usage
----

To use Crystal:

* Copy out the `crystal` directory into your project.
* Include Crystal
```
#include "crystal/crystal.cuh"
```
* Add the crystal directory to your include path

To run the operator implementations:

* Compile and run the operator. E.g.,
```
make bin/ops/project
./bin/ops/project
```

To run the Star Schema Benchmark implementation:

* Generate the test dataset

```
cd test/

# Generate the test generator / transformer binaries
cd ssb/dbgen
make
cd ../loader
make 
cd ../../

# Generate the test data and transform into columnar layout
# Substitute <SF> with appropriate scale factor (eg: 1)
python util.py ssb <SF> gen
python util.py ssb <SF> transform
```

* Configure the benchmark settings
```
cd src/ssb/
# Edit SF and BASE_PATH in ssb_utils.h
```

* To run a query, say run q11
```
make bin/ssb/q11
./bin/ssb/q11
```

