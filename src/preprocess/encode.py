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
        encoded_names = [f'{name}_{i}' for i in range(bits)]
        return encoded_names, encoded_values

    def fit(self, dataset):
        x = dataset.drop(['Dst Port', 'Protocol', 'Label'], axis=1)
        self.names = x.columns
        x = np.log(2 + x.to_numpy(dtype='float32'), dtype='float32')
        self.standard_scalar.fit(x[:1])
        self.standard_scalar.std_ = x.std(0)
        self.standard_scalar.mean_ = x.mean(0)

    def transform(self, dataset):
        y = dataset.pop('Label')
        x = dataset      
        dst_port_names, dst_port_values = Encoder.__to_binary('Dst Port', 16, x['Dst Port'].values)
        protocol_names, protocol_values = Encoder.__to_binary('Protocol', 8, x['Protocol'].values)
        x_t = x.drop(['Dst Port', 'Protocol'], axis=1).astype('float32')
        x_t = np.log(2 + x_t)
        x_t = self.standard_scalar.transform(x_t)
        x_t = pd.DataFrame(x_t, columns=self.names)
        x_t[dst_port_names] = dst_port_values
        x_t[protocol_names] = protocol_values
        x_t = x_t.assign(Label = y.values)
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
    train = pd.read_pickle(source_dir / "train.pickle")
    test = pd.read_pickle(source_dir / "test.pickle")
except Exception as e:
    print(f"Error occured while reading files {e}")
    sys.exit(-1)

e = Encoder()
e.fit(train)

dest_dir = pathlib.Path(args.dest_dir)
if not dest_dir.exists():
    print(f"Creating directory {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

train = e.transform(train)
test = e.transform(test)

# Label mapping to integers
labels = ["Benign","Bot","Brute Force -Web","Brute Force -XSS","DDOS attack-HOIC","DDOS attack-LOIC-UDP","DDoS attacks-LOIC-HTTP","DoS attacks-GoldenEye","DoS attacks-Hulk","DoS attacks-SlowHTTPTest","DoS attacks-Slowloris","FTP-BruteForce","Infilteration","SQL Injection","SSH-Bruteforce"]
label_map = {label: i for i, label in enumerate(labels)}

train['Label'] = train['Label'].map(label_map).astype('int8')
test['Label'] = test['Label'].map(label_map).astype('int8')

print("Saving encoded data")
train.to_csv(dest_dir / "train.csv")
test.to_csv(dest_dir/ "test.csv")

print("Encoding DONE")
