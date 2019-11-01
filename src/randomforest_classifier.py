import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize': (13.7, 10.27)})

parser = argparse.ArgumentParser(description="Train Random Forest Classifier on split dataset")
parser.add_argument('--source-dir', dest='source_dir', help="Path to split dataset")
parser.add_argument('--dest-file', dest='dest_file', help="File to store saved model")
parser.add_argument('--report-dir', dest='report_dir', help="Directory to store reports after testing")

args = parser.parse_args()
print("Starting Random Forest Classifier")

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
    print(f"Error occurred while reading files {e}")
    sys.exit(-1)

print("Training Random Forest Classifier")
r = RandomForestClassifier()
r.fit(x_train, y_train.values)

dest_file = pathlib.Path(args.dest_file)
if not dest_file.parent.exists():
    print(f"Creating directory {dest_file.parent}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
print("Saving Random Forest Classifier")
joblib.dump(r, dest_file)

print("Testing Random Forest Classifier")
report_dir = pathlib.Path(args.report_dir)
if not report_dir.exists():
    print(f"Creating directory {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
print("Testing classifier")
y_pred = r.predict(x_test)
labels = ['Benign', 'Bot', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk',
          'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection',
          'Infilteration', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris',
          'FTP-BruteForce', 'SSH-Bruteforce', 'DDOS attack-LOIC-UDP',
          'DDOS attack-HOIC']
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Saving reports")
sns.heatmap(cm, yticklabels=labels, xticklabels=labels, annot=True, fmt='.2f') \
    .set_title("(Normalized) Confusion Matrix for Random Forest Classifier on NSL-IDS 2018")
plt.savefig(report_dir / "random_forest_cm.png")
with open(report_dir / "report.txt", "w") as f:
    report = classification_report(y_test, y_pred, labels)
    print(report)
    f.write(report)
print("Random forest classifier DONE")
