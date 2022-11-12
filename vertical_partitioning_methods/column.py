import cost_model

def partition(workload):
    partitions = [[i+1] for i in range(workload.attribute_num)]
    
    return cost_model.calculate_cost(partitions,workload), partitions