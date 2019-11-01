import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Encoder:
    def __init__(self):
        self.standard_scalar = StandardScaler(copy=False)
        self.standard_scalar.check_is_fitted = lambda y: True
        self.names = []
        self._mu = 0
        self._sigma = 0

    @staticmethod
    def __to_binary(name, bits, values):
        encoded_values = pd.DataFrame((((values[:, None] & (1 << np.arange(bits)))) > 0).astype(int))
        encoded_names = [f'{name} {i}' for i in range(bits)]
        return encoded_names, encoded_values

    def fit(self, x):
        x = x.drop(['Dst Port', 'Protocol'], axis=1)
        self.names = x.columns
        x = np.log(2 + x.to_numpy())
        self.standard_scalar.fit(x[:1])
        self.standard_scalar.std_ = x.std(0)
        self.standard_scalar.mean_ = x.mean(0)

    def transform(self, x):
        dst_port_names, dst_port_values = Encoder.__to_binary('Dst Port', 16, x['Dst Port'].values)
        protocol_names, protocol_values = Encoder.__to_binary('Protocol', 8, x['Protocol'].values)
        x_t = x.drop(['Dst Port', 'Protocol'], axis=1)
        x_t = np.log(2 + x_t.to_numpy())
        x_t = self.standard_scalar.transform(x_t)
        x_t = pd.DataFrame(x_t, columns=self.names)
        x_t[dst_port_names] = dst_port_values
        x_t[protocol_names] = protocol_values
        return x_t


parser = argparse.ArgumentParser(description="Encode the split dataset")
parser.add_argument('--source-dir', dest='source_dir', help="Path to the directory containing split dataset")
parser.add_argument('--dest-dir', dest='dest_dir', help="Directory to store encoded dataset")
args = parser.parse_args()

source_dir = pathlib.Path(args.source_dir)

if not source_dir.exists():
    print(f"Error : {source_dir} does not exist")
    sys.exit(-1)

print("Reading files")
try:
    x_train = pd.read_pickle(source_dir / "x_train.pickle")
    x_test = pd.read_pickle(source_dir / "x_test.pickle")
    y_train = pd.read_pickle(source_dir / "y_train.pickle")
    y_test = pd.read_pickle(source_dir / "y_test.pickle")
except Exception as e:
    print(f"Error occured while reading files {e}")
    sys.exit(-1)

e = Encoder()
e.fit(x_train)

dest_dir = pathlib.Path(args.dest_dir)
if not dest_dir.exists():
    print(f"Creating directory {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

print("Saving encoded data")
x_train = e.transform(x_train)
x_train.to_pickle(dest_dir / "x_train")
x_test = e.transform(x_test)
x_test.to_pickle(dest_dir / "x_test")
y_train.to_pickle(dest_dir / "y_train")
y_test.to_pickle(dest_dir / "y_test")
y_train = pd.get_dummies(y_train)
y_train.to_pickle(dest_dir / "y_train_onehot")
y_test = pd.get_dummies(y_test)
y_test.to_pickle(dest_dir / "y_test_onehot")
print("Encoding DONE")
