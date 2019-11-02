import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

sns.set(rc={'figure.figsize': (13.7, 10.27)})


class Model:
    def __init__(self, conv_kernel_sizes=[8, 4], conv_kernel_count=[100, 64], intermediate_dense_size=32):
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_kernel_count = conv_kernel_count
        self.intermediate_dense_size = intermediate_dense_size
        self.input_shape = (100, 1)
        self.output_neurons = 14
        self.model = tf.keras.models.Sequential()

    def build(self):
        [self.model.add(layer) for layer in [
            tf.keras.layers.Conv1D(
                self.conv_kernel_count[0],
                self.conv_kernel_sizes[0],
                activation='relu',
                input_shape=self.input_shape
            ),
            # tf.keras.layers.MaxPooling1D(3),
            tf.keras.layers.Conv1D(
                self.conv_kernel_count[1],
                self.conv_kernel_sizes[0],
                activation='relu',
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(
                self.intermediate_dense_size,
                activation='relu'
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                self.output_neurons,
                activation='softmax'
            )
        ]
         ]
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def summary(self):
        return self.model.summary()

    def train(self, X_train, Y_train, model_save_path, validation_split=0.2, epochs=30):
        model_name = 'model_log_scaled_' + '1_no_max_pool_'.join(
            map(str, [*self.conv_kernel_count, *self.conv_kernel_sizes, self.intermediate_dense_size]))
        early_stopper = tf.keras.callbacks.EarlyStopping(min_delta=1, mode='min', monitor='val_loss', verbose=1,
                                                         patience=2)
        model_checkpointer = tf.keras.callbacks.ModelCheckpoint(model_save_path,
                                                                mode='min', monitor='val_loss', save_best_only=True)
        p = np.random.permutation(len(X_train))
        X_train = X_train[p]
        Y_train = Y_train[p]
        result = self.model.fit(X_train, Y_train, batch_size=16, epochs=epochs,
                                callbacks=[early_stopper, model_checkpointer], validation_split=validation_split)
        return result

    def predict(self, X):
        return self.model.predict(X)


parser = argparse.ArgumentParser(description="Train CNN on encoded dataset")
parser.add_argument('--source-dir', dest='source_dir', help="Path to encoded dataset")
parser.add_argument('--dest-file', dest='dest_file', help="File to store saved model")
parser.add_argument('--report-dir', dest='report_dir', help="Directory to store reports after testing")
parser.add_argument('--old-model', dest='old_model', help="Model to use if exists", default=None)

args = parser.parse_args()
print("Starting CNN Classifier")

source_dir = pathlib.Path(args.source_dir)

if not source_dir.exists():
    print(f"Error : {source_dir} does not exist")
    sys.exit(-1)

print("Reading files")
try:
    x_train = pd.read_pickle(source_dir / "x_train.pickle").to_numpy()
    x_test = pd.read_pickle(source_dir / "x_test.pickle").to_numpy()
    y_train = pd.read_pickle(source_dir / "y_train_onehot.pickle").to_numpy()
    y_test = pd.read_pickle(source_dir / "y_test_onehot.pickle").to_numpy()
except Exception as e:
    print(f"Error occurred while reading files {e}")
    sys.exit(-1)

report_dir = pathlib.Path(args.report_dir)
if not report_dir.exists():
    print(f"Creating directory {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
dest_file = pathlib.Path(args.dest_file)

model = Model()
epochs = 30
if args.old_model:
    old_model = pathlib.Path(args.old_model)
    if not old_model.exists():
        print("Old model does not exist...Error")
        sys.exit(-1)
    print("Reusing old model")
    epochs = 1
    model.model = tf.keras.models.load_model(old_model)
else:
    model.build()
print("Model Summary")
model.summary()
print("Training CNN Classifier")
x_train = x_train.reshape(*x_train.shape, 1)
if not dest_file.parent.exists():
    print(f"Creating directory {dest_file.parent}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
res = model.train(x_train, y_train, str(dest_file), epochs=epochs)
plt.figure()
plt.plot(res.history['accuracy'], label='accuracy')
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_accuracy'], label='validation accuracy')
plt.plot(res.history['val_loss'], label='validation_loss')
plt.legend()
plt.savefig(report_dir / "training_curve.png")
plt.close()

print("Testing CNN Classifier")

labels = np.array(['Benign', 'Bot', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk',
                   'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection',
                   'Infilteration', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris',
                   'FTP-BruteForce', 'SSH-Bruteforce', 'DDOS attack-LOIC-UDP',
                   'DDOS attack-HOIC'])

y_pred = model.predict(x_test.reshape(*x_test.shape, 1))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, yticklabels=labels, xticklabels=labels, annot=True, fmt='.2f').set_title(
    "(Normalized) Confusion Matrix for CNN Classifier on Log Scaled, Standardized NSL-IDS 2018")
plt.savefig(report_dir / "cnn_cm.png")
with open(report_dir / "report.txt", "w") as f:
    report = classification_report(labels[np.argmax(y_test, axis=1)], labels[np.argmax(y_pred, axis=1)])
    print(report)
    f.write(report)
print("CNN classifier DONE")
