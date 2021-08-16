import pandas as pd

def environmental_conditions(lines):
    pass

# df = pd.read_csv('/Users/Tom/Box/NCS Project/models/stochastic_model/CONTAM_data/test.txt', sep='\t', header=[3,4], index_col=[0,1])
with open('/Users/Tom/Box/NCS Project/models/stochastic_model/CONTAM_data/test2.txt', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print(f'{i}:', line)
    breakpoint()