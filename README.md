# Learning Vertical Partitions with Graph AutoEncoder

## Introduction

This is a PyTorch implementation of our proposed vertical partitioning methods: **VPGAE** and **VPGAE-B**, as described in our paper:

**VPGAE: Learning Vertical Partitions with Graph AutoEncoder**

## Requirements

- PyTorch (>=1.4.0)
- PyTorch Geometric
- numpy
- matplotlib
- sklearn
- more_itertools
- scipy
- tqdm
- pyvis
- psycopg2
- copy

We did not implement NAVATHE and O2P in our project because these two methods have been implemented in [Vertical partitioning algorithms used in physical design of databases](https://github.com/palatinuse/database-vertical-partitioning). We run this project and hard-code the partitioning results of NAVATHE and O2P in our files.