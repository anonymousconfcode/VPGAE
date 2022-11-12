import random

def synthetic_dataset(
    num = 100, 
    q_num_range = [5,50], 
    a_num_range = [1,100], 
    freq_range = [1,10], 
    lenth_of_attr_range = [2,50],
    cardinality_range = [1000,25000]
):
    dataset = []

    for data_id in range(num):
        data = {
            "table_info":{
                "name": "table{}".format(data_id)
            },
            "workload_info":{}
        }
        num_of_queries = random.randint(q_num_range[0],q_num_range[1])
        num_of_attributes = random.randint(a_num_range[0],a_num_range[1])
        length_of_attributes = []
        for i in range(num_of_attributes):
            length_of_attributes.append(random.randint(lenth_of_attr_range[0],lenth_of_attr_range[1]))
        
        referenced_attributes = []
        frequency_of_queries = []

        attribute_list = [i for i in range(1,num_of_attributes+1)]
        
        # for each query, select referenced_attributes and frequency_of_queries
        for query_id in range(num_of_queries):
            num_of_selected_attr = random.randint(1,max(1,int(num_of_attributes/num_of_queries)))
            random.shuffle(attribute_list)
            random_selected_attr = attribute_list[:num_of_selected_attr]
            random_selected_attr = sorted(random_selected_attr)
            referenced_attributes.append(random_selected_attr)
            
            if random.random() < 0.9:
                frequency_of_queries.append(random.randint(freq_range[0],freq_range[1]))
            else:
                frequency_of_queries.append(random.randint(30,100))
        
        data["table_info"]["num_of_attributes"] = num_of_attributes
        data["table_info"]["length_of_attributes"] = length_of_attributes
        data["table_info"]["cardinality"] = random.randint(cardinality_range[0],cardinality_range[1])
        
        data["workload_info"]["num_of_queries"] = num_of_queries
        data["workload_info"]["referenced_attributes"] = referenced_attributes
        data["workload_info"]["frequency_of_queries"] = frequency_of_queries

        dataset.append(data)
    
    return dataset

# TPC-H benckmark
def tpch_workload(scaleFactor):
    # table "customer"
    customer = {
        "table_info":{
            "name": "customer",
            "num_of_attributes": 8,
            "length_of_attributes": [4,25,40,4,15,4,10,117],
            "cardinality": scaleFactor * 150000
        },
        "workload_info":{
            "num_of_queries": 8,
            "referenced_attributes": [[1,7],[1,4],[1,4],[1,4],[1,2,3,4,5,6,8],[1],[1,2],[1,5,6]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1]
        }
    }
    # table "lineitem"
    lineitem = {
        "table_info":{
            "name": "lineitem",
            "num_of_attributes": 16,
            "length_of_attributes": [4,4,4,4,4,4,4,4,1,1,10,10,10,25,10,44],
            "cardinality": scaleFactor * 6000000
        },
        "workload_info":{
            "num_of_queries": 17,
            "referenced_attributes": [[5, 6, 7, 8, 9, 10, 11],[1,6,7,11],[1, 12, 13],[1, 3, 6, 7],[5, 6, 7, 11],[1, 3, 6, 7, 11],[1, 2, 3, 6, 7],[1, 2, 3, 5, 6, 7],[1, 6, 7, 9],[1, 11, 12, 13, 15],[2, 6, 7, 11],[3, 6, 7, 11],[2, 5, 6],[1, 5],[2, 5, 6, 7, 14, 15],[2, 3, 5, 11],[1, 3, 12, 13]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        }
    }
    
    # table "part"
    part = {
        "table_info":{
            "name": "part",
            "num_of_attributes": 9,
            "length_of_attributes": [4,55,25,10,25,4,10,4,23],
            "cardinality": scaleFactor * 200000
        },
        "workload_info":{
            "num_of_queries": 8,
            "referenced_attributes": [[1, 3, 5, 6],[1,5],[1,2],[1,5],[1,4,5,6],[1,4,7],[1,4,6,7],[1,2]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1]
        }
    }
 
    # table "supplier"
    supplier = {
        "table_info":{
            "name": "supplier",
            "num_of_attributes": 7,
            "length_of_attributes": [4,25,40,4,15,4,101],
            "cardinality": scaleFactor * 10000
        },
        "workload_info":{
            "num_of_queries": 10,
            "referenced_attributes": [[1,2,3,4,5,6,7],[1,4],[1,4],[1,4],[1,4],[1,4],[1,2,3,5],[1,7],[1,2,3,4],[1,2,4]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1,1,1]
        }
    }
    
    # table "partsupp" 
    partsupp = {
        "table_info":{
            "name": "partsupp",
            "num_of_attributes": 5,
            "length_of_attributes": [4,4,4,4,199],
            "cardinality": scaleFactor * 800000
        },
        "workload_info":{
            "num_of_queries": 5,
            "referenced_attributes": [[1,2,4],[1,2,4],[1,2,3,4],[1,2],[1,2,3]],
            "frequency_of_queries": [1,1,1,1,1]
        }
    }

    # table "orders"
    orders = {
        "table_info":{
            "name": "orders",
            "num_of_attributes": 9,
            "length_of_attributes": [4,4,1,4,10,15,15,4,79],
            "cardinality": scaleFactor * 1500000
        },
        "workload_info":{
            "num_of_queries": 12,
            "referenced_attributes": [[1,2,5,8],[1,5,6],[1,2,5],[1,2],[1,2,5],[1,5],[1, 2, 5],[1,6],[1,2,9],[1,2,4,5],[1,3],[2]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1,1,1,1,1]
        }
    }
    
    # table "nation"
    nation = {
        "table_info":{
            "name": "nation",
            "num_of_attributes": 4,
            "length_of_attributes": [4,25,4,152],
            "cardinality": 25
        },
        "workload_info":{
            "num_of_queries": 9,
            "referenced_attributes": [[1,2,3],[1,2,3],[1,2],[1,2,3],[1,2],[1,2],[1,2],[1,2],[1,2]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1,1]
        }
    }

    # table "region"
    region = {
        "table_info":{
            "name": "region",
            "num_of_attributes": 3,
            "length_of_attributes": [4,25,152],
            "cardinality": 5
        },
        "workload_info":{
            "num_of_queries": 3,
            "referenced_attributes": [[1,2],[1,2],[1,2]],
            "frequency_of_queries": [1,1,1]
        }
    }
    
    
    return [customer,lineitem,orders,supplier,part,partsupp,nation,region]

# TPC-DS benchmark with scaleFactor = 1
def tpcds_workload():
    customer_address = {
        "table_info":{
            "name": "customer_address",
            "num_of_attributes": 13,
            "length_of_attributes": [4, 17, 11, 9, 16, 11, 9, 14, 3, 11, 14, 5, 21],
            "cardinality": 50000
        },
        "workload_info":{
            "num_of_queries": 6,
            "referenced_attributes": [[1, 9, 11], [1, 9, 10], [1, 10], [1, 9, 11], [1, 7], [1, 12]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1]
        }
    }

    customer_demographics = {
        "table_info":{
            "name": "customer_demographics",
            "num_of_attributes": 9,
            "length_of_attributes": [4, 2, 2, 21, 4, 11, 4, 4, 4],
            "cardinality": 1920800
        },
        "workload_info":{
            "num_of_queries": 7,
            "referenced_attributes": [[1, 2, 3, 4], [1, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 3, 4], [1], [1, 3, 4]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1]
        }
    }

    date_dim = {
        "table_info":{
            "name": "date_dim",
            "num_of_attributes": 28,
            "length_of_attributes": [4, 17, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10, 7, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2],
            "cardinality": 73049
        },
        "workload_info":{
            "num_of_queries": 23,
            "referenced_attributes": [[1, 3, 7, 9], [1, 3, 7], [1, 3, 7], [1, 3, 7, 11], [1, 3, 7, 9], [1, 3, 4], [1, 3, 7], [1, 3, 7], [1, 3, 7, 10], [1, 3, 7, 9], [1, 3, 7, 15], [1, 3, 7], [1, 3, 7, 9], [1, 3, 4, 11], [1, 3, 7, 9], [1, 3, 4], [1, 3, 4, 9], [1, 3, 4, 7, 9, 11], [1, 3, 7, 10], [1, 3, 7, 8], [1, 3, 7, 9], [1, 3, 7, 9], [1, 3, 4]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    }

    warehouse = {
        "table_info":{
            "name": "warehouse",
            "num_of_attributes": 14,
            "length_of_attributes": [4, 17, 18, 4, 11, 9, 16, 11, 9, 18, 3, 11, 14, 5],
            "cardinality": 5
        },
        "workload_info":{
            "num_of_queries": 2,
            "referenced_attributes": [[1, 3], [1, 3]],
            "frequency_of_queries": [1, 1]
        }
    }
    
    ship_mode = {
        "table_info":{
            "name": "ship_mode",
            "num_of_attributes": 6,
            "length_of_attributes": [4, 17, 31, 11, 21, 21],
            "cardinality": 20
        },
        "workload_info":{
            "num_of_queries": 2,
            "referenced_attributes": [[1, 3], [1, 3]],
            "frequency_of_queries": [1, 1]
        }
    }

    time_dim = {
        "table_info":{
            "name": "time_dim",
            "num_of_attributes": 10,
            "length_of_attributes": [4, 17, 4, 4, 4, 4, 3, 21, 21, 21],
            "cardinality": 86400
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 3, 4, 5]],
            "frequency_of_queries": [1]
        }
    }

    reason = {
        "table_info":{
            "name": "reason",
            "num_of_attributes": 3,
            "length_of_attributes": [4, 17, 101],
            "cardinality": 35
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 3]],
            "frequency_of_queries": [1]
        }
    }

    income_band = {
        "table_info":{
            "name": "income_band",
            "num_of_attributes": 3,
            "length_of_attributes": [4, 4, 4],
            "cardinality": 20
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 2, 3]],
            "frequency_of_queries": [1]
        }
    }

    item = {
        "table_info":{
            "name": "item",
            "num_of_attributes": 22,
            "length_of_attributes": [4, 17, 4, 4, 102, 6, 6, 4, 51, 4, 51, 4, 51, 4, 51, 21, 21, 21, 11, 11, 4, 51],
            "cardinality": 18000
        },
        "workload_info":{
            "num_of_queries": 13,
            "referenced_attributes": [[1, 8, 9, 14, 15], [1, 2], [1, 8, 9, 14, 15, 21], [1, 9, 11, 13, 22], [1, 2], [1, 2], [1, 12, 13, 21], [1, 8, 9, 21], [1, 9, 11, 13, 14, 15], [1, 8, 9, 21], [1, 9, 11, 13, 21], [1, 9, 11, 13, 22], [1, 9, 11, 13]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    }
    
    store = {
        "table_info":{
            "name": "store",
            "num_of_attributes": 29,
            "length_of_attributes": [4, 17, 4, 4, 4, 5, 4, 4, 21, 13, 4, 8, 56, 15, 4, 8, 4, 8, 3, 7, 16, 11, 7, 18, 3, 11, 14, 5, 5],
            "cardinality": 12
        },
        "workload_info":{
            "num_of_queries": 13,
            "referenced_attributes": [[1], [1, 26], [1, 25], [1, 24], [1, 2, 6, 28], [1], [1], [1], [1, 2], [1, 24], [1, 7, 23], [1, 6, 18], [1, 6]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    }

    call_center = {
        "table_info":{
            "name": "call_center",
            "num_of_attributes": 31,
            "length_of_attributes": [4, 17, 4, 4, 4, 4, 12, 6, 4, 4, 21, 13, 4, 51, 70, 13, 4, 4, 4, 51, 11, 11, 16, 11, 7, 18, 3, 11, 14, 5, 5],
            "cardinality": 6
        },
        "workload_info":{
            "num_of_queries": 2,
            "referenced_attributes": [[1, 2, 7, 12], [1, 7]],
            "frequency_of_queries": [1, 1]
        }
    }

    customer = {
        "table_info":{
            "name": "customer",
            "num_of_attributes": 18,
            "length_of_attributes": [4, 17, 4, 4, 4, 4, 4, 11, 21, 31, 2, 4, 4, 4, 9, 56, 51, 4],
            "cardinality": 100000
        },
        "workload_info":{
            "num_of_queries": 7,
            "referenced_attributes": [[1, 5], [1, 5], [1, 8, 9, 10, 11], [1, 8, 9, 10, 11], [1, 9, 10], [2, 3, 4, 5, 9, 10], [1, 3, 4, 5]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1]
        }
    }

    web_site = {
        "table_info":{
            "name": "web_site",
            "num_of_attributes": 26,
            "length_of_attributes": [4, 17, 4, 4, 7, 4, 4, 8, 13, 4, 33, 69, 13, 4, 51, 11, 10, 16, 11, 7, 18, 3, 11, 14, 5, 4],
            "cardinality": 30
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 5]],
            "frequency_of_queries": [1]
        }
    }

    store_returns = {
        "table_info":{
            "name": "store_returns",
            "num_of_attributes": 20,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "cardinality": 287514
        },
        "workload_info":{
            "num_of_queries": 2,
            "referenced_attributes": [[5], [3, 9, 10, 11]],
            "frequency_of_queries": [1, 1]
        }
    }

    household_demographics = {
        "table_info":{
            "name": "household_demographics",
            "num_of_attributes": 5,
            "length_of_attributes": [4, 4, 16, 4, 4],
            "cardinality": 7200
        },
        "workload_info":{
            "num_of_queries": 7,
            "referenced_attributes": [[1, 4], [1, 3, 4, 5], [1, 3, 4, 5], [1, 4, 5], [1, 2], [1, 3], [1, 4]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1]
        }
    }

    web_page = {
        "table_info":{
            "name": "web_page",
            "num_of_attributes": 14,
            "length_of_attributes": [4, 17, 4, 4, 4, 4, 2, 4, 19, 51, 4, 4, 4, 4],
            "cardinality": 60
        },
        "workload_info":{
            "num_of_queries": 0,
            "referenced_attributes": [],
            "frequency_of_queries": []
        }
    }

    promotion = {
        "table_info":{
            "name": "promotion",
            "num_of_attributes": 19,
            "length_of_attributes": [4, 17, 4, 4, 4, 5, 4, 51, 2, 2, 2, 2, 2, 2, 2, 2, 41, 16, 2],
            "cardinality": 60
        },
        "workload_info":{
            "num_of_queries": 2,
            "referenced_attributes": [[1, 10, 15], [1, 10, 15]],
            "frequency_of_queries": [1, 1]
        }
    }

    catalog_page = {
        "table_info":{
            "name": "catalog_page",
            "num_of_attributes": 9,
            "length_of_attributes": [4, 17, 4, 4, 11, 4, 4, 75, 8],
            "cardinality": 11718
        },
        "workload_info":{
            "num_of_queries": 0,
            "referenced_attributes": [],
            "frequency_of_queries": []
        }
    }

    inventory = {
        "table_info":{
            "name": "inventory",
            "num_of_attributes": 4,
            "length_of_attributes": [4, 4, 4, 4],
            "cardinality": 11745000
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 2, 4]],
            "frequency_of_queries": [1]
        }
    }

    catalog_returns = {
        "table_info":{
            "name": "catalog_returns",
            "num_of_attributes": 27,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "cardinality": 144067
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 8, 12, 27]],
            "frequency_of_queries": [1]
        }
    }

    web_returns = {
        "table_info":{
            "name": "web_returns",
            "num_of_attributes": 24,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "cardinality": 71763
        },
        "workload_info":{
            "num_of_queries": 0,
            "referenced_attributes": [],
            "frequency_of_queries": []
        }
    }

    web_sales = {
        "table_info":{
            "name": "web_sales",
            "num_of_attributes": 34,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 6, 3, 6, 6, 6, 7, 7, 6],
            "cardinality": 719384
        },
        "workload_info":{
            "num_of_queries": 1,
            "referenced_attributes": [[1, 3, 14, 15, 16]],
            "frequency_of_queries": [1]
        }
    }

    catalog_sales = {
        "table_info":{
            "name": "catalog_sales",
            "num_of_attributes": 34,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 6, 3, 6, 6, 6, 7, 7, 6],
            "cardinality": 1441548
        },
        "workload_info":{
            "num_of_queries": 3,
            "referenced_attributes": [[1, 4, 22], [1, 5, 16, 17, 19, 21, 22, 28], [1, 3, 12, 14, 15]],
            "frequency_of_queries": [1, 1, 1]
        }
    }
    
    store_sales = {
        "table_info":{
            "name": "store_sales",
            "num_of_attributes": 23,
            "length_of_attributes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 3, 6, 6, 7, 6, 3, 6, 6, 6],
            "cardinality": 2880404
        },
        "workload_info":{
            "num_of_queries": 19,
            "referenced_attributes": [[1, 3, 14], [1, 3, 5, 9, 11, 13, 14, 20], [1, 5, 6, 7, 8, 11, 14, 16, 17, 23], [1, 3, 4, 8, 16], [1, 3, 5, 8, 11, 13, 14, 20], [1, 4, 6, 8, 10], [1, 3, 16], [1, 8, 14], [1, 5, 7, 8, 11, 14, 23], [1, 3, 16], [1, 3, 8, 14], [1, 3, 16], [1, 3, 8, 14], [1, 3, 8, 11, 14], [1, 4, 6, 8, 10], [1, 4, 6, 7, 8, 10, 20, 23], [1, 3, 8, 14], [3, 4, 10, 11, 14], [2, 6, 8]],
            "frequency_of_queries": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    }
    
    return [customer_address,customer_demographics,date_dim,warehouse,ship_mode,time_dim,reason,income_band,
        item,store,call_center,customer,web_site,store_returns,household_demographics,web_page,promotion,
        catalog_page,inventory,catalog_returns,web_returns,web_sales,catalog_sales,store_sales]

# number of queries, number of attrs, referenced attrs, frequency, selectivity, length of attrs, scan key, clustered index, cardinality
def real_system_wide_table():
    wide_table = {
        "table_info":{
            "name": "wide_table",
            "num_of_attributes": 30,
            "length_of_attributes": [150,500,233,300,4,50,4,100,4,4,500,100,250,4,4,1000,300,25,4,400,4,33,100,55,155,4,4,900,20,4],
            "cardinality": 100158
        },
        "workload_info":{
            "num_of_queries": 10,
            "referenced_attributes": [[5, 6, 7, 8, 9, 10, 11],[1, 12, 13],[1,6,7,11],[1, 3, 6, 7],[5, 6, 7, 11],[1, 4],[17,20,21,22],[18,19,23,27],[22,25,26,27,28,29,30],[2,14,15,16,24]],
            "frequency_of_queries": [1,1,1,1,1,1,1,1,1,1]
        }
    }

    return wide_table