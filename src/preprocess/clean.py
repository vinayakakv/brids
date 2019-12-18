import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Clean NSL-IDS 2018 Dataset")
parser.add_argument('--source-file', dest='source_file', help="Path to original dataset")
parser.add_argument('--dest-file', dest='dest_file', help="Path to store cleaned dataset")

args = parser.parse_args()
try:
    dtypes = {'Dst Port': np.dtype('int32'), 'Protocol': np.dtype('int32'), 'Timestamp': np.dtype('O'), 'Flow Duration': np.dtype('int32'), 'Tot Fwd Pkts': np.dtype('int32'), 'Tot Bwd Pkts': np.dtype('int32'), 'TotLen Fwd Pkts': np.dtype('int32'), 'TotLen Bwd Pkts': np.dtype('float32'), 'Fwd Pkt Len Max': np.dtype('int32'), 'Fwd Pkt Len Min': np.dtype('int32'), 'Fwd Pkt Len Mean': np.dtype('float32'), 'Fwd Pkt Len Std': np.dtype('float32'), 'Bwd Pkt Len Max': np.dtype('int32'), 'Bwd Pkt Len Min': np.dtype('int32'), 'Bwd Pkt Len Mean': np.dtype('float32'), 'Bwd Pkt Len Std': np.dtype('float32'), 'Flow Byts/s': np.dtype('float32'), 'Flow Pkts/s': np.dtype('float32'), 'Flow IAT Mean': np.dtype('float32'), 'Flow IAT Std': np.dtype('float32'), 'Flow IAT Max': np.dtype('float32'), 'Flow IAT Min': np.dtype('float32'), 'Fwd IAT Tot': np.dtype('float32'), 'Fwd IAT Mean': np.dtype('float32'), 'Fwd IAT Std': np.dtype('float32'), 'Fwd IAT Max': np.dtype('float32'), 'Fwd IAT Min': np.dtype('float32'), 'Bwd IAT Tot': np.dtype('float32'), 'Bwd IAT Mean': np.dtype('float32'), 'Bwd IAT Std': np.dtype('float32'), 'Bwd IAT Max': np.dtype('float32'), 'Bwd IAT Min': np.dtype('float32'), 'Fwd PSH Flags': np.dtype('int32'), 'Bwd PSH Flags': np.dtype('int32'), 'Fwd URG Flags': np.dtype('int32'), 'Bwd URG Flags': np.dtype('int32'), 'Fwd Header Len': np.dtype('int32'), 'Bwd Header Len': np.dtype('int32'), 'Fwd Pkts/s': np.dtype('float32'), 'Bwd Pkts/s': np.dtype('float32'), 'Pkt Len Min': np.dtype('int32'), 'Pkt Len Max': np.dtype('int32'), 'Pkt Len Mean': np.dtype('float32'), 'Pkt Len Std': np.dtype('float32'), 'Pkt Len Var': np.dtype('float32'), 'FIN Flag Cnt': np.dtype('int32'), 'SYN Flag Cnt': np.dtype('int32'), 'RST Flag Cnt': np.dtype('int32'), 'PSH Flag Cnt': np.dtype('int32'), 'ACK Flag Cnt': np.dtype('int32'), 'URG Flag Cnt': np.dtype('int32'), 'CWE Flag Count': np.dtype('int32'), 'ECE Flag Cnt': np.dtype('int32'), 'Down/Up Ratio': np.dtype('int32'), 'Pkt Size Avg': np.dtype('float32'), 'Fwd Seg Size Avg': np.dtype('float32'), 'Bwd Seg Size Avg': np.dtype('float32'), 'Fwd Byts/b Avg': np.dtype('int32'), 'Fwd Pkts/b Avg': np.dtype('int32'), 'Fwd Blk Rate Avg': np.dtype('int32'), 'Bwd Byts/b Avg': np.dtype('int32'), 'Bwd Pkts/b Avg': np.dtype('int32'), 'Bwd Blk Rate Avg': np.dtype('int32'), 'Subflow Fwd Pkts': np.dtype('int32'), 'Subflow Fwd Byts': np.dtype('int32'), 'Subflow Bwd Pkts': np.dtype('int32'), 'Subflow Bwd Byts': np.dtype('int32'), 'Init Fwd Win Byts': np.dtype('int32'), 'Init Bwd Win Byts': np.dtype('int32'), 'Fwd Act Data Pkts': np.dtype('int32'), 'Fwd Seg Size Min': np.dtype('int32'), 'Active Mean': np.dtype('float32'), 'Active Std': np.dtype('float32'), 'Active Max': np.dtype('float32'), 'Active Min': np.dtype('float32'), 'Idle Mean': np.dtype('float32'), 'Idle Std': np.dtype('float32'), 'Idle Max': np.dtype('float32'), 'Idle Min': np.dtype('float32'), 'Label': np.dtype('O')}
    df = pd.read_csv(args.source_file, dtype=dtypes, na_values=['Infinity'], memory_map=True)
except Exception as e:
    print(f"Could not open {args.source_file} for reading. Error str(e)")
    sys.exit(-1)

print("Start cleaning dataset")
df[df.columns[17]] = df[df.columns[17]].astype('float32')
df[df.columns[18]] = df[df.columns[18]].astype('float32')
df = df.drop([df.columns[0], 'Timestamp'], axis=1)
df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
df = df[df['Flow Duration'] >= 0]
df = df[df['Flow IAT Min'] >= 0]
dest_file = pathlib.Path(args.dest_file)
if not dest_file.parent.exists():
    print(f"Creating directory {dest_file.parent}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
print("Writing cleaned dataset to pickle")
df.to_pickle(dest_file)
print("Data Cleaning DONE")
