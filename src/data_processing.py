import luigi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Encoder:
    def __init__(self):
        self.standard_scalar = StandardScaler(copy=False)
        self.standard_scalar.check_is_fitted = lambda y: True
        self.names = []
        self._mu = 0
        self._sigma = 0

    @staticmethod
    def __to_binary(name, bits, values):
        encoded_values = pd.DataFrame(((values[:, None] & (1 << np.arange(bits))) > 0).astype(int))
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


class DataCleaningTask(luigi.Task):
    """
    Data Cleaning Task specific to NSL-IDS 2018 dataset
    Drops negative Flow durations and converts NaN and Inf to -1
    """
    dataset_name = luigi.Parameter(default="combined")
    visualize = luigi.BoolParameter(default=False)

    def requires(self):
        return None

    def run(self):
        print("Running Data Cleaning Tasks")
        df = pd.read_csv(self.input().path)
        df[df.columns[17]] = df[df.columns[17]].astype('float32')
        df[df.columns[18]] = df[df.columns[18]].astype('float32')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
        df = df.drop([df.columns[0], 'Timestamp'], axis=1)
        df.to_csv(self.output().path, index=False)
        if self.visualize:
            yield DataVisualizationTask(prefix="cleaned_data", dataset_location=self.output().path)

    def input(self):
        return luigi.LocalTarget(path="data/" + self.dataset_name + '.csv')

    def output(self):
        return luigi.LocalTarget(path="data/" + self.dataset_name + '_cleaned.csv')


class DataSplitTask(luigi.Task):
    """
    Splits the dataset into test and train parts and saves them
    """
    split_ratio = luigi.FloatParameter(default=0.2)

    def requires(self):
        return DataCleaningTask()

    def run(self):
        df = pd.read_csv(self.input().path)
        x = df.loc[:, df.columns != 'Label']
        y = df['Label']
        del df
        sss = StratifiedShuffleSplit(1, test_size=self.split_ratio, random_state=0)
        split_indices = list(sss.split(x, y))[0]
        x_train = x.iloc[split_indices[0]]
        y_train = y.iloc[split_indices[0]]
        x_test = x.iloc[split_indices[1]]
        y_test = y.iloc[split_indices[1]]
        del x, y
        x_train.to_csv(self.output()["x_train"].path, index=False)
        x_test.to_csv(self.output()["x_test"].path, index=False)
        y_train.to_csv(self.output()["y_train"].path, index=False)
        y_test.to_csv(self.output()["y_test"].path, index=False)

    def output(self):
        return {
            "x_train": luigi.LocalTarget(path="data/x_train.csv"),
            "x_test": luigi.LocalTarget(path="data/x_test.csv"),
            "y_train": luigi.LocalTarget(path="data/y_train.csv"),
            "y_test": luigi.LocalTarget(path="data/y_test.csv")
        }


class DataNormalizationTask(luigi.Task):
    """
    Log transform and standardize dataset
    """
    # output_file_prefix = luigi.Parameter(default="normalized")
    one_hot_encode_y = luigi.BoolParameter()
    visualize = luigi.BoolParameter(default=False)

    def requires(self):
        return DataSplitTask()

    def output(self):
        return {
            "x_train": luigi.LocalTarget(path="data/normalized_x_train.csv"),
            "x_test": luigi.LocalTarget(path="data/normalized_x_test.csv"),
            "y_train": luigi.LocalTarget(path="data/normalized_y_train.csv"),
            "y_test": luigi.LocalTarget(path="data/normalized_y_test.csv")
        }

    def run(self):
        x_train = pd.read_csv(self.input()["x_train"].path)
        x_test = pd.read_csv(self.input()["x_test"].path)
        encoder = Encoder()
        encoder.fit(x_train)
        x_train = encoder.transform(x_train)
        x_test = encoder.transform(x_test)
        x_train.to_csv(self.output()["x_train"].path, index=False)
        x_test.to_csv(self.output()["x_test"].path, index=False)
        if self.one_hot_encode_y:
            y_train = pd.read_csv(self.input()["y_train"])
            y_test = pd.read_csv(self.input()["y_test"])
            y_train = pd.get_dummies(y_train)
            y_test = pd.get_dummies(y_test)
            y_train.to_csv(self.output()["y_train"].path, index=False)
            y_test.to_csv(self.output()["y_test"].path, index=False)
        if self.visualize:
            yield DataVisualizationTask(prefix="normalized_data", dataset_location=self.output().path)


class DataVisualizationTask(luigi.Task):
    """
    Visualizes data
    Independent module as of itself -- can be called from any node which yields csv of dataset
    Draws boxplot of all attributes (except label)
    """
    sns.set(rc={'figure.figsize': (13.7, 10.27)})
    prefix = luigi.Parameter()
    report_location = "report/dataset/%s/" % prefix
    dataset_location = luigi.Parameter()

    def requires(self):
        return None

    def input(self):
        return luigi.LocalTarget(path=self.dataset_location)

    def run(self):
        sns.set(rc={'figure.figsize': (13.7, 10.27)})
        df = pd.read_csv(self.input().path)
        cols = list(df)
        with self.output().open('w') as log:
            for col in cols[2:len(cols) - 1]:
                print(f"Col {col}")
                try:
                    plt.figure()
                    sns.boxplot(x=df[col], y=df['Label'])
                    plt.savefig(self.report_location + f"{col.replace(' ', '_').replace('/', ' per ')}_boxplot.png")
                except Exception as e:
                    print(f"Error {e}")
            log.write(f"Plotted {col}")

    def output(self):
        return luigi.LocalTarget(path=self.report_location + "DataVisualizationTask" + self.prefix + ".log")


if __name__ == '__main__':
    # luigi.build([DataPreprocessTask()], local_scheduler=True)
    luigi.run()
