import argparse
import pathlib
import sys

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser(description="Split Dataset into training and testing parts")
parser.add_argument('--source-file', dest='source_file', help="Path to cleaned dataset")
parser.add_argument('--dest-dir', dest='dest_dir', help="Directory to store split dataset")
parser.add_argument('--ratio', dest='ratio', help="Ratio of the testing data in the split", default=0.2, type=float)

args = parser.parse_args()

try:
    df = pd.read_pickle(args.source_file)
except Exception as e:
    print(f"Could not open {args.src_file} for reading. Error str(e)")
    sys.exit(-1)
print("Reading the dataset")
x = df.loc[:, df.columns != 'Label']
y = df['Label']
del df
print("Splitting the dataset to train and test parts")
sss = StratifiedShuffleSplit(1, test_size=args.ratio, random_state=0)
split_indices = list(sss.split(x, y))[0]
x_train = x.iloc[split_indices[0]]
y_train = y.iloc[split_indices[0]]
x_test = x.iloc[split_indices[1]]
y_test = y.iloc[split_indices[1]]
print(f"Writing the split dataset to {args.dest_dir}")
dest_dir = pathlib.Path(args.dest_dir)
if not dest_dir.exists():
    print(f"Creating directory {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
x_train.to_pickle(dest_dir / "x_train.pickle")
x_test.to_pickle(dest_dir / "x_test.pickle")
y_train.to_pickle(dest_dir / "y_train.pickle")
y_test.to_pickle(dest_dir / "y_test.pickle")
print("Data splitting DONE")
