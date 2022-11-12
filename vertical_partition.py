import time

import cost_model
import argparse
import numpy as np
import random
import torch

from data import dataset
from vertical_partitioning_methods import VPGAE
from vertical_partitioning_methods import column
from vertical_partitioning_methods import row
from vertical_partitioning_methods import optimal
from vertical_partitioning_methods import hillclimb
from metrics.unnecessary_data_read import fraction_of_unnecessary_data_read
from metrics.reconstruction_joins import number_of_joins
from data.workload_class import VPGAE_Workload, Workload

seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed = 777) # CPU pytorch seed

parser = argparse.ArgumentParser(description='VPGAE')
parser.add_argument('--dataset', default="tpc_h", type=str, \
    help='Options: tpc_h, tpc_ds, and synthetic.')


def print_costs_func(dataset_, costs):
    for data, cost in zip(dataset_, costs):
        if cost != "OOT":
            print("{}: {}".format(data["table_info"]["name"], round(cost,2)))
        else:
            print("{}: {}".format(data["table_info"]["name"], "OOT"))
    print()

# TPC-H benchmark experiments
def tpc_h_exp():
    use_OPTIMAL = False
    tpc_h = dataset.tpch_workload(10)

    beam_costs = []
    kmeans_costs = []
    hill_costs = []
    column_costs = []
    row_costs = []
    optimal_costs = []

    beam_partitions_list = []
    kmeans_partitions_list = []
    hill_partitions_list = []
    column_partitions_list = []
    row_partitions_list = []
    optimal_partitions_list = []

    workload_list = []

    for data in tpc_h:
        workload = Workload(data)
        vpgae_workload = VPGAE_Workload(data)

        num_node = vpgae_workload.affinity_matrix.shape[0]

        beam_cost, beam_partitions = VPGAE.partition(
            algo_type="VPGAE-B",
            workload=vpgae_workload, 
            n_hid=num_node, 
            n_dim=2*num_node, 
            k=2, 
            origin_candidate_length=6, 
            beam_search_width=3
        )
        kmeans_cost, kmeans_partitions = VPGAE.partition(
            algo_type="VPGAE",
            workload=vpgae_workload, 
            n_hid=num_node, 
            n_dim=2*num_node,
            k=2
        )
        
        hill_cost, hill_partitions = hillclimb.partition(workload=workload)
        column_cost, column_partitions = column.partition(workload=workload)
        row_cost, row_partitions = row.partition(workload=workload)
        if use_OPTIMAL:
            optimal_cost, optimal_partitions = optimal.partition(workload=workload)
            optimal_costs.append(optimal_cost)
            optimal_partitions_list.append(optimal_partitions)
        
        beam_costs.append(beam_cost)
        kmeans_costs.append(kmeans_cost)
        hill_costs.append(hill_cost)
        column_costs.append(column_cost)
        row_costs.append(row_cost)
        
        beam_partitions_list.append(beam_partitions)
        kmeans_partitions_list.append(kmeans_partitions)
        hill_partitions_list.append(hill_partitions)
        column_partitions_list.append(column_partitions)
        row_partitions_list.append(row_partitions)

        workload_list.append(workload)

    navathe_partitions_list = [
        [[1], [2], [7, 3], [4], [6, 5], [8]],
        [[1], [2], [3], [4], [5], [7, 6], [8], [9], [16, 10], [11], [13, 12], [14], [15]],
        [[1], [2], [7, 3], [4], [5], [6], [8], [9]],
        [[1], [2], [3], [4], [5], [7, 6]],
        [[1], [2], [9, 3], [4], [5], [6], [7], [8]],
        [[3, 2, 1], [5, 4]],
        [[3, 2, 1], [4]],
        [[2, 1], [3]]
    ]

    navathe_costs = [cost_model.calculate_cost(partitions, workload) \
        for partitions, workload in zip(navathe_partitions_list, workload_list)]

    o2p_partitions_list = [
        [[1], [2], [7, 3], [4], [5], [6], [8]],
        [[1], [5, 2], [3], [4], [7, 6], [8], [9], [16, 10], [11], [12], [13], [14], [15]],
        [[1], [2], [7, 6, 3], [8, 4], [5], [9]],
        [[1], [2], [3], [4], [5], [7, 6]],
        [[1], [2], [9, 3], [7, 4], [5], [6], [8]],
        [[3, 2, 1], [5, 4]],
        [[3, 2, 1], [4]],
        [[2, 1], [3]]
    ]
    
    o2p_costs = [cost_model.calculate_cost(partitions, workload) \
        for partitions, workload in zip(o2p_partitions_list, workload_list)]

    print("VPGAE-B's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, beam_costs)
    print("VPGAE's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, kmeans_costs)
    print("HILLCLIMB's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, hill_costs)
    print("COLUMN's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, column_costs)
    print("ROW's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, row_costs)
    print("NAVATHE's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, navathe_costs)
    print("O2P's estimated scan costs on 8 tables:")
    print_costs_func(tpc_h, o2p_costs)

    if use_OPTIMAL:
        print("OPTIMAL costs on 8 tables:", optimal_costs)
    
    print("Unnecessary data read of VPGAE-B:", fraction_of_unnecessary_data_read(beam_partitions_list, workload_list))
    print("Unnecessary data read of VPGAE:", fraction_of_unnecessary_data_read(kmeans_partitions_list, workload_list))
    print("Unnecessary data read of HILLCLIMB:", fraction_of_unnecessary_data_read(hill_partitions_list, workload_list))
    print("Unnecessary data read of COLUMN:", fraction_of_unnecessary_data_read(column_partitions_list, workload_list))
    print("Unnecessary data read of ROW:", fraction_of_unnecessary_data_read(row_partitions_list, workload_list))
    print("Unnecessary data read of NAVATHE:", fraction_of_unnecessary_data_read(navathe_partitions_list, workload_list))
    print("Unnecessary data read of O2P:", fraction_of_unnecessary_data_read(o2p_partitions_list, workload_list))

    if use_OPTIMAL:
        print("Unnecessary data read of OPTIMAL:", fraction_of_unnecessary_data_read(optimal_partitions_list, workload_list))
    print()

    column_RJ = np.sum(number_of_joins(column_partitions_list, workload_list))
    if column_RJ == 0:
        print("column reconstruction joins = 0")
    else:
        print("Normalized reconstruction joins of VPGAE-B:", np.sum(number_of_joins(beam_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of VPGAE:", np.sum(number_of_joins(kmeans_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of HILLCLIMB:", np.sum(number_of_joins(hill_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of COLUMN:", column_RJ/column_RJ)
        print("Normalized reconstruction joins of ROW:", np.sum(number_of_joins(row_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of NAVATHE:", np.sum(number_of_joins(navathe_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of O2p:", np.sum(number_of_joins(o2p_partitions_list, workload_list))/column_RJ)
        
        if use_OPTIMAL:
            print("Normalized reconstruction joins of OPTIMAL:", np.sum(number_of_joins(optimal_partitions_list, workload_list))/column_RJ)

    print("--------------------")

# TPC-DS benchmark experiments
def tpc_ds_exp():
    tpc_ds = dataset.tpcds_workload()
    beam_costs = []
    kmeans_costs = []
    hill_costs = []
    column_costs = []
    row_costs = []
    
    beam_partitions_list = []
    kmeans_partitions_list = []
    hill_partitions_list = []
    column_partitions_list = []
    row_partitions_list = []
    workload_list = []

    for data in tpc_ds:
        workload = Workload(data)
        vpgae_workload = VPGAE_Workload(data)
        
        num_node = vpgae_workload.affinity_matrix.shape[0]

        beam_cost, beam_partitions = VPGAE.partition(
            algo_type="VPGAE-B", 
            workload=vpgae_workload, 
            n_hid=num_node, 
            n_dim=2*num_node, 
            k=2, 
            origin_candidate_length=6, 
            beam_search_width=3
        )
        kmeans_cost, kmeans_partitions = VPGAE.partition(
            algo_type="VPGAE", 
            workload=vpgae_workload, 
            n_hid=num_node, 
            n_dim=2*num_node, 
            k=2
        )
        
        hill_cost, hill_partitions = hillclimb.partition(workload=workload)
        column_cost, column_partitions = column.partition(workload=workload)
        row_cost, row_partitions = row.partition(workload=workload)

        beam_costs.append(beam_cost)
        kmeans_costs.append(kmeans_cost)
        hill_costs.append(hill_cost)
        column_costs.append(column_cost)
        row_costs.append(row_cost)

        beam_partitions_list.append(beam_partitions)
        kmeans_partitions_list.append(kmeans_partitions)
        hill_partitions_list.append(hill_partitions)
        column_partitions_list.append(column_partitions)
        row_partitions_list.append(row_partitions)
        workload_list.append(workload)
    
    navathe_partitions_list = [
        [[1], [2], [3], [4], [5], [6], [7], [8], [11, 9], [10], [13, 12]],
        [[1], [9, 4, 3, 2], [5], [6], [7], [8]],
        "OOT",
        [[13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [14]],
        [[5, 4, 3, 2, 1], [6]],
        [[1], [2], [5, 4, 3], [6], [7], [8], [9], [10]],
        [[3, 1], [2]],
        [[3, 2, 1]],
        [[1], [2], [3], [4], [5], [6], [7], [21, 9, 8], [10], [13, 11], [20, 12], [15, 14], [16], [17], [18], [19], [22]],
        "OOT",
        "OOT",
        [[1], [2], [18, 3], [5, 4], [6], [7], [11, 8], [10, 9], [12], [13], [14], [15], [16], [17]],
        "OOT",
        [[1], [2], [11, 10, 9, 3], [4], [5], [6], [7], [8], [12], [13], [14], [15], [16], [17], [18], [19], [20]],
        [[5, 4, 3, 2, 1]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
        [[18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [19]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [[4, 2, 1], [3]],
        "OOT",
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
        "OOT",
        "OOT",
        [[1], [2], [3], [4], [11, 5], [6], [23, 7], [8], [22, 9], [10], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21]]
    ]
    
    navathe_costs = []
    for partitions, workload in zip(navathe_partitions_list, workload_list):
        if partitions != "OOT":
            navathe_costs.append(cost_model.calculate_cost(partitions, workload))
        else:
            navathe_costs.append("OOT")

    o2p_partitions_list = [
        [[1], [8, 6, 5, 4, 3, 2], [7], [11, 9], [10], [13, 12]],
        [[1], [9, 4, 3, 2], [8, 7, 6, 5]],
        [[1], [15, 2], [3], [11, 4], [5], [6], [7], [28, 8], [9], [10], [12], [13], [14], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27]],
        [[13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [14]],
        [[5, 4, 3, 2, 1], [6]],
        [[1], [2], [3], [5, 4], [6], [7], [8], [9], [10]],
        [[3, 1], [2]],
        [[2, 1], [3]],
        [[1], [2], [3], [4], [5], [6], [7], [21, 9, 8], [10], [13, 11], [20, 12], [15, 14], [16], [17], [18], [19], [22]],
        [[28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1], [29, 7]],
        [[30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [31, 12]],
        [[1], [2], [18, 3], [4], [5], [6], [7], [11, 8], [10, 9], [12], [13], [14], [15], [16], [17]],
        [[25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [26]],
        [[19, 18, 17, 16, 15, 14, 13, 12, 8, 7, 6, 1], [20, 2], [11, 10, 9, 3], [5, 4]],
        [[5, 4, 1], [3, 2]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
        [[18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [19]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [[4, 2, 1], [3]],
        [[1], [27, 2], [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 7, 6, 5, 4, 3], [12, 8], [26]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
        [[1], [2], [14, 3], [21, 20, 19, 18, 17, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4], [16, 15], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34]],
        [[1], [2], [3], [34, 4], [28, 21, 19, 17, 16, 5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [18], [20], [22], [23], [24], [25], [26], [27], [29], [30], [31], [32], [33]],
        [[1], [2], [3], [4], [5], [6], [7], [8], [22, 9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [23]]
    ]

    o2p_costs = [cost_model.calculate_cost(partitions, workload) \
        for partitions, workload in zip(o2p_partitions_list, workload_list)]

    print("VPGAE-B's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, beam_costs)
    print("VPGAE's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, kmeans_costs)
    print("HILLCLIMB's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, hill_costs)
    print("COLUMN's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, column_costs)
    print("ROW's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, row_costs)
    print("NAVATHE's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, navathe_costs)
    print("O2P's estimated scan costs on 24 tables:")
    print_costs_func(tpc_ds, o2p_costs)
    
    print("Unnecessary data read of VPGAE-B:", fraction_of_unnecessary_data_read(beam_partitions_list, workload_list))
    print("Unnecessary data read of VPGAE:", fraction_of_unnecessary_data_read(kmeans_partitions_list, workload_list))
    print("Unnecessary data read of HILLCLIMB:", fraction_of_unnecessary_data_read(hill_partitions_list, workload_list))
    print("Unnecessary data read of COLUMN:", fraction_of_unnecessary_data_read(column_partitions_list, workload_list))
    print("Unnecessary data read of ROW:", fraction_of_unnecessary_data_read(row_partitions_list, workload_list))
    # print("Unnecessary data read of NAVATHE:", fraction_of_unnecessary_data_read(navathe_partitions_list, workload_list))
    print("Unnecessary data read of O2P:", fraction_of_unnecessary_data_read(o2p_partitions_list, workload_list))

    print()
    column_RJ = np.sum(number_of_joins(column_partitions_list, workload_list))
    if column_RJ == 0:
        print("column reconstruction joins = 0")
    else:
        print("Normalized reconstruction joins of VPGAE-B:", np.sum(number_of_joins(beam_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of VPGAE:", np.sum(number_of_joins(kmeans_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of HILLCLIMB:", np.sum(number_of_joins(hill_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of COLUMN:", column_RJ/column_RJ)
        print("Normalized reconstruction joins of ROW:", np.sum(number_of_joins(row_partitions_list, workload_list))/column_RJ)
        # print("Normalized reconstruction joins of NAVATHE:", np.sum(number_of_joins(navathe_partitions_list, workload_list))/column_RJ)
        print("Normalized reconstruction joins of O2P:", np.sum(number_of_joins(o2p_partitions_list, workload_list))/column_RJ)

    print("--------------------")

# synthetic dataset experiments
def synthetic_dataset_exp():
    attributes_num_list = [75, 100, 125, 150, 175]
    for num_of_attributes in attributes_num_list:
        print("tables have {} attributes.".format(num_of_attributes))
        synthetic_database = dataset.synthetic_dataset(
            num = 10, 
            a_num_range = [num_of_attributes, num_of_attributes]
        )
        
        beam_costs = []
        kmeans_costs = []
        hill_costs = []
        column_costs = []
        
        beam_times = []
        kmeans_times = []
        hill_times = []
        
        # process each table and its workload
        for data in synthetic_database:
            workload = Workload(data)
            vpgae_workload = VPGAE_Workload(data)
            num_node = vpgae_workload.affinity_matrix.shape[0]

            t2=time.time()
            kmeans_cost, kmeans_partitions = VPGAE.partition(
                algo_type="VPGAE", 
                workload=vpgae_workload, 
                n_hid=num_node, 
                n_dim=2*num_node, 
                k=2
            )
            kmeans_time=time.time()-t2
            print("VPGAE cost:{}, time:{:.3f}".format(kmeans_cost,kmeans_time))

            t1=time.time()
            beam_cost, beam_partitions = VPGAE.partition(
                algo_type="VPGAE-B", 
                workload=vpgae_workload, 
                n_hid=num_node, 
                n_dim=2*num_node, 
                k=2, 
                origin_candidate_length=6, 
                beam_search_width=3
            )
            # beam_cost, beam_partitions = 0,[]
            beam_time=time.time()-t1
            print("VPGAE-B cost:{}, time:{:.3f}".format(beam_cost,beam_time))
            
            t3=time.time()
            hill_cost, hill_partitions = hillclimb.partition(workload=workload)
            # hill_cost, hill_partitions = 0,[]
            hill_time=time.time()-t3
            print("HILLCLIMB cost:{}, time:{:.3f}".format(hill_cost,hill_time))
            print("")

            column_cost, column_partitions = column.partition(workload=workload)

            beam_costs.append(beam_cost)
            kmeans_costs.append(kmeans_cost)
            
            hill_costs.append(hill_cost)
            column_costs.append(column_cost)

            beam_times.append(beam_time)
            kmeans_times.append(kmeans_time)
            
            hill_times.append(hill_time)

        print("Avg. VPGAE cost:{}".format(np.mean(kmeans_costs)))
        print("Avg. VPGAE-B cost:{}".format(np.mean(beam_costs)))
        print("Avg. HILLCLIMB cost:{}".format(np.mean(hill_costs)))
        
        print("Avg. VPGAE time:{}".format(np.mean(kmeans_times)))
        print("Avg. VPGAE-B time:{}".format(np.mean(beam_times)))
        print("Avg. HILLCLIMB time:{}".format(np.mean(hill_times)))
        print("--------------------")

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset
    if dataset_name == "tpc_h":
        tpc_h_exp()
    elif dataset_name == "tpc_ds":
        tpc_ds_exp()
    elif dataset_name == "synthetic":
        synthetic_dataset_exp()
    else:
        raise ValueError("No such dataset or benchmark, available options: \n1.tpc_h \n2.tpc_ds \n3.synthetic")