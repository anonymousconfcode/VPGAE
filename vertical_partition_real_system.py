import time
import psycopg2
import copy
import cost_model
import argparse
import numpy as np
import random
import torch

seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed = 777) # CPU pytorch seed

from data import dataset
from vertical_partitioning_methods import VPGAE
from vertical_partitioning_methods import column
from vertical_partitioning_methods import row
from vertical_partitioning_methods import optimal
from vertical_partitioning_methods import hillclimb
from data.workload_class import VPGAE_Workload, Workload

parser = argparse.ArgumentParser(description='A toy experiment of vertical partitioning on postgres.')
parser.add_argument('--method', default="VPGAE", type=str, \
    help='Options: VPGAE-B, VPGAE, hillclimb, column, row, o2p, navathe.')

wide_table_attrs = [["a1","CHAR(150) NOT NULL"],
    ["a2","CHAR(500) NOT NULL"],
    ["a3","CHAR(233) NOT NULL"],
    ["a4","CHAR(300) NOT NULL"],
    ["a5","INTEGER NOT NULL"],
    ["a6","CHAR(50) NOT NULL"],
    ["a7","INTEGER NOT NULL"],
    ["a8","CHAR(100) NOT NULL"],
    ["a9","INTEGER NOT NULL"],
    ["a10","INTEGER NOT NULL"],
    ["a11","CHAR(500) NOT NULL"],
    ["a12","CHAR(100) NOT NULL"],
    ["a13","CHAR(250) NOT NULL"],
    ["a14","INTEGER NOT NULL"],
    ["a15","INTEGER NOT NULL"],
    ["a16","CHAR(1000) NOT NULL"],
    ["a17","CHAR(300) NOT NULL"],
    ["a18","CHAR(25) NOT NULL"],
    ["a19","INTEGER NOT NULL"],
    ["a20","CHAR(400) NOT NULL"],
    ["a21","INTEGER NOT NULL"],
    ["a22","CHAR(33) NOT NULL"],
    ["a23","CHAR(100) NOT NULL"],
    ["a24","CHAR(55) NOT NULL"],
    ["a25","CHAR(155) NOT NULL"],
    ["a26","INTEGER NOT NULL"],
    ["a27","INTEGER NOT NULL"],
    ["a28","CHAR(900) NOT NULL"],
    ["a29","CHAR(20) NOT NULL"],
    ["a30","INTEGER NOT NULL"]]


def do_partitioning_on_pg(partitions,workload):
    subtables = ["wide_table"+str(i) for i in range(len(partitions))]

    conn = psycopg2.connect(database="wide_test", user="postgres", password="lhy19990316", host="127.0.0.1", port="5432")
    conn.autocommit = True
    cursor = conn.cursor()

    st = time.time()
    temp_partitions = copy.deepcopy(partitions)
    
    
    # Create subtables 
    for index,subtable in enumerate(subtables):
        print(subtable+":")
        for attrid in temp_partitions[index]:
            print(wide_table_attrs[attrid-1][0])

        sql = "create table "+subtable+" ("

        for attrid in temp_partitions[index]:
            sql += wide_table_attrs[attrid-1][0] + " " + wide_table_attrs[attrid-1][1] + ","

        sql = sql + "tuple_id INTEGER NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 ))"
        cursor.execute(sql)

    # Insert data into subtables
    for idx,subtable in enumerate(subtables):
        sql = "insert into " + subtable + " select "
        for attrid in temp_partitions[idx]:
            sql += wide_table_attrs[attrid-1][0] + ","
        sql = sql[:-1]
        
        sql += " from wide_table"
        cursor.execute(sql)
    print("layout time: {}".format(time.time()-st))
    
    sql_list = []
    # Execute all queries in the workload
    for index,query_attributes in enumerate(workload.referenced_attributes):
        # Find all subtables that are referenced in the query
        dict_ = dict()

        for idx,partition in enumerate(partitions):
            for attrid in query_attributes:
                if attrid in partition:
                    if idx in list(dict_.keys()):
                        dict_[idx].append(attrid)
                    else:
                        dict_[idx] = [attrid]
        temp_sql_list = []
        for partitionid, attr_list in dict_.items():
            sql = "explain analyse select "
            for attr in attr_list:
                sql += wide_table_attrs[attr-1][0] + ","
            sql = sql[:-1]
            sql += " from " +  "wide_table" + str(partitionid)
            temp_sql_list.append(sql)
        
        for i in range(int(workload.freq[index])):
            for temp_sql in temp_sql_list:
                sql_list.append(temp_sql)
        print(temp_sql_list)
        print("frequency = "+str(int(workload.freq[index])))
        print("----------------------------")

    print("start cache warm-up.")
    # cache warm-up
    for _ in range(5):
        for sql in sql_list:
            cursor.execute(sql)
            # print(cursor.fetchall())
    print("end cache warm-up.")

    pg_execution_time_list = []
    for _ in range(5):
        pg_execution_time = 0
        for sql in sql_list:
            cursor.execute(sql)
            # print(cursor.fetchall()[-1][0].split(" ")[2])
            pg_execution_time += eval(cursor.fetchall()[-1][0].split(" ")[2])
        pg_execution_time_list.append(pg_execution_time)
    
    print("real execution time of workload on partitioned tables: {} (returned by pg).".format(np.mean(pg_execution_time_list)/1000))
    
    # Delete all subtables
    for subtable in subtables:
        sql = "drop table " + subtable + ";"
        cursor.execute(sql)
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    args = parser.parse_args()
    data = dataset.real_system_wide_table()
    workload = Workload(data)
    vpgae_workload = VPGAE_Workload(data)

    if args.method == "VPGAE-B":
        cost, partitions = VPGAE.partition(
            algo_type="VPGAE-B",
            workload=vpgae_workload,
            n_hid=32,
            n_dim=16,
            k=3,
            origin_candidate_length=3,
            beam_search_width=3
        )
    elif args.method == "VPGAE":
        cost, partitions = VPGAE.partition(
            algo_type="VPGAE",
            workload=vpgae_workload,
            n_hid=32,
            n_dim=16,
            k=3
        )
    elif args.method == "hillclimb":
        cost, partitions = hillclimb.partition(workload=workload)
    elif args.method == "column":
        cost, partitions = column.partition(workload=workload)
    elif args.method == "row":
        cost, partitions = row.partition(workload=workload)
    elif args.method == "o2p":
        partitions = [[27, 19, 18, 1], [2], [3], [23, 4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [21, 20, 17], [22], [24], [30, 29, 28, 26, 25]]
        cost = cost_model.calculate_cost(partitions,workload)
    elif args.method == "navathe":
        print("NAVATHE is out of time.") 
    else:
        raise ValueError("Invalid vertical partitioning method name.")

    if args.method != "navathe":
        partitions=sorted(partitions,key=lambda x:min(x))
        
        print("This is {}.".format(args.method))
        do_partitioning_on_pg(partitions,workload)
        print("Estimated scan cost:", cost)
        print("---------------------------")