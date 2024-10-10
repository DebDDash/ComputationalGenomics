import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str,
                    required=True, help="Image_Name")
args = parser.parse_args()

image_name = args.image
if 'merfish' in image_name:
    dataset_i = 'MERFISH'
else:
    dataset_i = 'OSMFISH'


df = pd.read_csv(
    f'Step3_Output_{image_name}/TCNLabel_MajorityVoting.csv', header=None, names=['Label'])
df['ID'] = df.index
df = df[['ID', 'Label']]
df.to_csv(f'CytoCommunity_{dataset_i}.csv', index=False)
