""" utility script: when copying results back from CX1 this combines
the run model log to the master model log
"""

import argparse
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Append model log to master model log.')
parser.add_argument('model_log', type=str,)
args = parser.parse_args()
try:
    model_log = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/{args.model_log}_model_log.csv',
                            index_col=0)
except FileNotFoundError:
    print('File not found. check spelling?')
    model_log = pd.DataFrame()
model_log['run name'] = args.model_log
model_log['backed up'] = False
try:
    master = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv',
                         index_col=0)
    if args.model_log in master['run name'].unique():
        answer = input(f'''models from run: {args.model_log} are already logged in the master file.
                       Do you want to continue to append the new log? [Y/N]''')
    else:
        answer ='y'

except FileNotFoundError:
    master = pd.DataFrame()
    answer = 'y'
    
if 'y' in answer.lower():
    master = master.append(model_log, verify_integrity=True)
    master.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')