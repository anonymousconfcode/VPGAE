from tqdm import tqdm
import cost_model
import more_itertools as mit

# optimal method
def partition(workload):
    best_cost = float('inf')
    best_partitioning_scheme = []
    lst = [i+1 for i in range(workload.attribute_num)]

    for k in range(1,len(lst)+1):
        for partition in tqdm(mit.set_partitions(lst,k)):
            temp_cost = cost_model.calculate_cost(partition,workload)
            if temp_cost < best_cost:
                best_cost = temp_cost
                best_partitioning_scheme = partition

    return best_cost, best_partitioning_scheme