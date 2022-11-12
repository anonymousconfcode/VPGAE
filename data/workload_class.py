import numpy as np
import copy

# Usage matrix to affinity matrix
def usage_matrix_to_affinity_matrix(usage_matrix):
    attribute_num = usage_matrix.shape[1] - 1
    query_num = usage_matrix.shape[0]

    affinity_matrix = np.zeros((attribute_num,attribute_num),dtype = np.float32)
    for i in range(attribute_num):
        for j in range(i,attribute_num):
            affinity_value = 0
            for z in range(query_num):
                if usage_matrix[z, i] == 1 and usage_matrix[z, j] == 1:
                    affinity_value += usage_matrix[z, -1]
            affinity_matrix[i][j] = affinity_value
            affinity_matrix[j][i] = affinity_value
    
    return affinity_matrix

# Define workload profile class for VPGAE and VPGAE-B
class VPGAE_Workload():
    def __init__(self, data):
        self.referenced_attributes = data["workload_info"]["referenced_attributes"]
        self.query_num = data["workload_info"]["num_of_queries"]
        self.attribute_num = data["table_info"]["num_of_attributes"]
        self.length_of_attributes = data["table_info"]["length_of_attributes"]
        self.cardinality = data["table_info"]["cardinality"]

        self.usage_matrix = np.zeros((self.query_num,self.attribute_num+1),dtype = np.float32)
        for query_id in range(self.query_num):
            for attr_id in self.referenced_attributes[query_id]:
                self.usage_matrix[query_id][attr_id-1] = 1.0
        
        self.subsets = self.init_subsets(self.attribute_num, self.referenced_attributes)
        # print(sets)
        # print(self.usage_matrix)

        delete_attr = []
        for subset in self.subsets:
            if len(subset) > 1:
                delete_attr += subset[1:]

        if len(delete_attr) != 0:
            self.new_usage_matrix = np.delete(self.usage_matrix, np.array(delete_attr)-1, axis = 1)
        else:
            self.new_usage_matrix = self.usage_matrix

        self.map_newindex_2_oldindex = {}
        
        for new_index in range(self.new_usage_matrix.shape[1]-1):
            old_index_list = []
            for old_index in range(self.usage_matrix.shape[1]-1):
                if False not in (self.new_usage_matrix[:,new_index] == self.usage_matrix[:,old_index]):
                    old_index_list.append(old_index+1)
                self.map_newindex_2_oldindex[new_index+1] = old_index_list

        self.freq = np.array(data["workload_info"]["frequency_of_queries"], dtype = np.float32)
        self.new_usage_matrix[:,-1] = self.freq
        self.affinity_matrix = usage_matrix_to_affinity_matrix(self.new_usage_matrix)

    def init_subsets(self, attribute_num, referenced_attributes):
        partitions = [[i+1 for i in range(attribute_num)]]
        for query_attr in referenced_attributes:
            temp_partitions = copy.deepcopy(partitions)
            for partition in temp_partitions:
                accessed_attrs = []
                ignored_attrs = []

                for attr in query_attr:
                    if attr in partition:
                        accessed_attrs.append(attr)
                for attr in partition:
                    if attr not in accessed_attrs:
                        ignored_attrs.append(attr)

                partitions.remove(partition)
                if len(accessed_attrs) > 0:
                    partitions.append(accessed_attrs)
                if len(ignored_attrs) > 0:
                    partitions.append(ignored_attrs)
        return partitions

# Define workload profile class for other baselines
class Workload():
    def __init__(self, data):
        self.referenced_attributes = data["workload_info"]["referenced_attributes"]
        self.query_num = data["workload_info"]["num_of_queries"]
        self.attribute_num = data["table_info"]["num_of_attributes"]
        self.length_of_attributes = data["table_info"]["length_of_attributes"]
        self.cardinality = data["table_info"]["cardinality"]

        self.usage_matrix = np.zeros((self.query_num,self.attribute_num+1),dtype = np.float32)
        for query_id in range(self.query_num):
            for attr_id in self.referenced_attributes[query_id]:
                self.usage_matrix[query_id][attr_id-1] = 1.0
        
        self.freq = np.array(data["workload_info"]["frequency_of_queries"], dtype = np.float32)
        self.usage_matrix[:,-1] = self.freq
        self.affinity_matrix = usage_matrix_to_affinity_matrix(self.usage_matrix)