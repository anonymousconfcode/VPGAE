import math

class SYS_PARAM():
    block_size = 8 * 1024
    buffer_size = 1024 * block_size
    seek_time = 0.008
    disk_bandwidth = 92 * 1024 * 1024

def calculate_cost(partitioning_scheme, workload):
    cost = 0

    # calculate the cost of each query
    for query_id in range(workload.query_num):
        # extract referenced partitions
        referenced_attributes = workload.referenced_attributes[query_id]
        referenced_partitions = []
        for attr in referenced_attributes:
            for partition in partitioning_scheme:
                if attr in partition and partition not in referenced_partitions:
                    referenced_partitions.append(partition)
        
        each_partition_row_size = []
        referenced_partitions_row_size = 0
        for partition in referenced_partitions:
            row_size = 0
            # calculate the row size of the partition
            for attr in partition:
                row_size += workload.length_of_attributes[attr-1]
            each_partition_row_size.append(row_size)
            referenced_partitions_row_size += row_size
        
        query_cost = 0
        # for each partition, calculate the cost
        for p_id, partition in enumerate(referenced_partitions):
            partition_row_size = each_partition_row_size[p_id]
            # we have to read at least one block from the disk
            partition_buffer_size = max(SYS_PARAM.block_size, \
                math.floor((SYS_PARAM.buffer_size * partition_row_size)/referenced_partitions_row_size))
            
            blocks_read_per_buffer = math.floor(partition_buffer_size / SYS_PARAM.block_size)
            # the total number of blocks of the current partition
            number_of_blocks = math.ceil((partition_row_size * workload.cardinality) / SYS_PARAM.block_size)

            seek_cost = SYS_PARAM.seek_time * math.ceil(number_of_blocks / blocks_read_per_buffer)
            scan_cost = number_of_blocks * SYS_PARAM.block_size / SYS_PARAM.disk_bandwidth

            current_cost = seek_cost + scan_cost
            query_cost += current_cost
        
        cost += workload.freq[query_id] * query_cost
    
    return cost