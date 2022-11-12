from data import dataset

tpc_ds = dataset.tpcds_workload()
wide_table = [dataset.real_system_wide_table()]

indent_str = "    "

for data in wide_table:
    table_func = indent_str + "public static Table tpchCustomer(BenchmarkConfig conf){\n"
    for attribute_id in range(data["table_info"]["num_of_attributes"]):
        table_func += indent_str + indent_str + "Attribute a{} = ".format(attribute_id) + "new Attribute(\"a{}\", AttributeType.CharacterVarying({}));\n".format(attribute_id, data["table_info"]["length_of_attributes"][attribute_id])

    table_func += indent_str + indent_str + "List<Attribute> attributes = new ArrayList<Attribute>();\n"

    for attribute_id in range(data["table_info"]["num_of_attributes"]):
        table_func += indent_str + indent_str + "attributes.add(a{});\n".format(attribute_id)

    table_func += indent_str + indent_str + "Table t = new Table(\"{}\", conf.getTableType(), attributes);\n".format(data["table_info"]["name"])
    table_func += indent_str + indent_str + "t.pk = \"a0\";\n"
    table_func += indent_str + indent_str + "t.workload = BenchmarkWorkloads.tpchCustomer(attributes, conf.getScaleFactor());\n"
    table_func += indent_str + indent_str + "t.workload.dataFileName = conf.getDataFileDir() + \"customer.tbl\";\n"
    table_func += indent_str + indent_str + "return t;\n"
    table_func += indent_str + "}"

    print(table_func)
    print("")
    print("")

for data in wide_table:
    workload_func = indent_str + "public static Workload tpchCustomer(List<Attribute> attributes, double scaleFactor){\n"
    workload_func += indent_str + indent_str + "Workload w = new Workload(attributes, {}, \"{}\");\n".format(data["table_info"]["cardinality"], data["table_info"]["name"])
    for query_id in range(data["workload_info"]["num_of_queries"]):
        workload_func += indent_str + indent_str + "w.addProjectionQuery(\"q{}\", {}".format(query_id, data["workload_info"]["frequency_of_queries"][query_id])
        for attr_id in data["workload_info"]["referenced_attributes"][query_id]:
            workload_func += ", {}".format(attr_id-1)
        workload_func += ");\n"
    
    workload_func += indent_str + indent_str + "return w;\n"
    workload_func += indent_str + "}"
    print(workload_func)
    
    print("")
    print("")