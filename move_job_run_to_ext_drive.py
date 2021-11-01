'''Utility script: will send results with a particular run_name
(created when ran in CX1) to an external hard drive.
TODO: Change the destination directory'''
import shutil
import argparse
import os
import time
import pprint
import pandas as pd

parser = argparse.ArgumentParser(description='Run name to move to external hard drive')
parser.add_argument('run_name', type=str,)
args = parser.parse_args()
DATA_LOC = f'{os.path.dirname(os.path.realpath(__file__))}/results'

try:
    model_log = pd.read_csv(f'{DATA_LOC}/model_log.csv', index_col=0)
except FileNotFoundError:
    print('File not found. check spelling?')
model_log.loc[model_log['run name'] == args.run_name, ['backed up']] = True


FILES_TO_MOVE_TO_EXT_DRIVE = model_log.loc[model_log['run name'] == args.run_name, 'model']
DRIVE = f'/mnt/usb-WD_Elements_2620_575832314441394E37445032-0:0-part1/NCS Project/stochastic_model/results'
for FILE in FILES_TO_MOVE_TO_EXT_DRIVE:
    print(f'Moving file: {FILE}')
    try:
        # os.path.join(src, file), os.path.join(dst, file)
        # copy file to the external harddrive
        shutil.move(os.path.join(DATA_LOC, f'{FILE}.pickle'),os.path.join(DRIVE, f'{FILE}.pickle'))

    except FileNotFoundError as e:
        print(e)
        print(f'file : {FILE}.pickle not found. Has it already been transferred to the external harddrive?')

model_log.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/results/model_log.csv')

