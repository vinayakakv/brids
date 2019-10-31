import argparse
import pandas as pd
import sys
import numpy as np
import pathlib

parser = argparse.ArgumentParser(description="Clean NSL-IDS 2018 Dataset")
parser.add_argument('--source-file', dest='source_file', help="Path to original dataset")
parser.add_argument('--dest-file', dest='dest_file', help="Path to store cleaned dataset")

args = parser.parse_args()
try:
    df = pd.read_csv(args.source_file)
except Exception as e:
    print(f"Could not open {args.src_file} for reading. Error str(e)")
    sys.exit(-1)

print("Start cleaning dataset")
df = df.drop([df.columns[0], 'Timestamp'], axis=1)
df[df.columns[17]] = df[df.columns[17]].astype('float32')
df[df.columns[18]] = df[df.columns[18]].astype('float32')
df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
df = df[df['Flow Duration'] >= 0]
df = df[df['Flow IAT Min'] >= 0]
dest_file = pathlib.Path(args.dest_file)
if not dest_file.parent.exists():
    print(f"Creating directory {dest_file.parent}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
print("Writing cleaned dataset to pickle")
df.to_pickle(dest_file)
print("CLEANING DONE")
