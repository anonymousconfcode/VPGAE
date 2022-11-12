input_str = \
'''
21 8
12
19
15
6
22
4
10
13
2
7
3
5
9
16
1
11
14
17
18
20
0
'''

partitions = []

for partition in input_str.split("\n"):
    if len(partition) == 0:
        continue

    partitions.append([eval(attr)+1 for attr in partition.split(" ")])

partitions = sorted(partitions,key=lambda x:min(x))
print(partitions)