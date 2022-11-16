# Learning Vertical Partitions with Graph AutoEncoder

## Introduction

This is the official implementation of our proposed vertical partitioning methods: **VPGAE** and **VPGAE-B**.

## Requirements
This project is based on Python3 and requires the following modules: 
- torch
- torch_geometric
- math
- sklearn
- more_itertools
- numpy
- argparse
- copy
- time
- tqdm
- random
- psycopg2

Before proceeding to the next step, please ensure your Python environment is successfully installed.

## Run our experiments
All experimental results reported in our paper can be easily reproduced as follows.

### Estimated cost and real cost (RQ1)
Research question 1 (RQ1) aims to validate the reliability of the cost model used in our paper. 

To run this experiment, you should install PostgreSQL 11.2 on your computer and create an empty database named "wide_test" in it. Then, for files `initialize_wide_table.py` and `vertical_partition_real_system.py`, you should modify the parameters (such as user, password, host, etc.) in the function `psycopg2.connect()` to ensure that the database connection is correct. Next, you need to run `python initialize_wide_table.py` to create a wide table and insert 100158 rows in it. Finally, you can simply run the following command to obtain the results of RQ1:
```
sh ./scripts/run_real_system.sh
```
By default, this script will use VPGAE as the vertical partitioning method. To see results of other methods, you can modify "--method" in file `./scripts/run_real_system.sh` and the optional methods are listed in `vertical_partition_real_system.py`.

### Performance on TPC-H and TPC-DS Benchmark (RQ2)
Research question 2 (RQ2) aims to examine the performance of our proposed methods and baselines on two benchmarks: TPC-H and TPC-DS. You can run the following two scripts to see the results on TPC-H and TPC-DS, respectively.
```
sh ./scripts/run_exp_on_tpc_h.sh
sh ./scripts/run_exp_on_tpc_ds.sh
```
### Performance on Very Wide Tables (RQ3)
Research question 3 (RQ3) aims to demonstrate the adaptivity of VPGAE(-B) on very large tables. You can simply run the following command to obtain the results of RQ3:
```
sh ./scripts/run_exp_on_synthetic_dataset.sh
```

**Attention:** 
- We didn't implement NAVATHE and O2P in our project because they have been implemented in this project: [Vertical partitioning algorithms used in physical design of databases](https://github.com/palatinuse/database-vertical-partitioning). We ran the above project on our datasets, and then hardcoded the partitioning results for NAVATHE and O2P in our project.
- Running RQ3 will take a lot of time because HILLCLIMB is too slow on very wide tables.
- Since Pytorch involves many floating-point calculations, the partitioning results from VPGAE(-B) may differ slightly on different machines.
