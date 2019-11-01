import seaborn as sns
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import pathlib

parser = argparse.ArgumentParser(description="Visualize the dataset as boxplot")
parser.add_argument('--source-file', dest='source_file', help="Path to dataset")
parser.add_argument('--dest-dir', dest='dest_dir', help="Directory to store results")
parser.add_argument('--disable-grouping', dest='no_group', help="Disables grouping of instances based on classes",
                    action='store_true')

args = parser.parse_args()
print("Starting Dataset Visualization")

try:
    df = pd.read_pickle(args.source_file)
except Exception as e:
    print(f"Could not open {args.src_file} for reading. Error str(e)")
    sys.exit(-1)

dest_dir = pathlib.Path(args.dest_dir)
if not dest_dir.exists():
    print(f"Creating directory {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

sns.set(rc={'figure.figsize': (13.7, 10.27)})
for col in list(df)[2:]:
    print(f"Col {col}")
    try:
        plt.figure()
        if args.no_group:
            sns.boxplot(x=df[col])
        else:
            sns.boxplot(x=df[col], y=df['Label'])
        plt.savefig(dest_dir / f"{col.replace(' ', '_').replace('/', ' per ')}_boxplot.png")
        plt.close()
    except Exception as e:
        print(f"Error {repr(e)}")
        continue
