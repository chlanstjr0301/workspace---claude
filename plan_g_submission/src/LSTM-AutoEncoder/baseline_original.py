# Official KAMP guidebook baseline.
# Do not modify this file for experiments.
# All ablations must copy this file and change exactly one factor.
# Source: docs/Guidebook_소성가공 예지보전 AI 데이터셋.pdf, codes 2-33.
# Only indentation, typographic quotes/minus signs, project I/O, and logging
# are adapted. Original computation, plotting expressions and defaults remain.

# Output plumbing required by this project; no training settings are changed.
import os
import sys
import json
import hashlib
import platform
import traceback
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('KAMP_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'baseline_original'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log = (OUTPUT_DIR / 'execution_log.txt').open('w', encoding='utf-8')

class _LogStream:
    # Mirror output without consuming randomness or touching the model.
    def __init__(self, stream):
        self.stream = stream
    def write(self, value):
        self.stream.write(value)
        _log.write(value)
        _log.flush()
    def flush(self):
        self.stream.flush()
        _log.flush()
    def isatty(self):
        return False

sys.stdout = _LogStream(sys.stdout)
sys.stderr = _LogStream(sys.stderr)

# [Code 2, PDF 23] Original library imports; no random seed is added.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model, models, layers, optimizers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sklearn

_environment = {'python': platform.python_version(), 'tensorflow': tf.__version__,
                'keras': tf.keras.__version__, 'pandas': pd.__version__,
                'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                'tensorflow_gpus': [str(g) for g in tf.config.list_physical_devices('GPU')],
                'random_seed': None, 'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS')}
print('Baseline Original environment:', json.dumps(_environment))
_baseline_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_inputs_sha256 = {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
                  for name in ['press_data_normal.csv', 'outlier_data.csv']}

# [Code 3, PDF 23] Load the original CSVs with their first column as index.
normal = pd.read_csv(DATA_DIR / 'press_data_normal.csv', index_col=0)
outlier = pd.read_csv(DATA_DIR / 'outlier_data.csv', index_col=0)

# [Codes 4-11, PDF 24-26] Original inspection and dataset copies.
print(normal.head())
print(outlier.head())
normal_data = normal.copy()
outlier_data = outlier.copy()
print('Normal Data 개수 : {}'.format(len(normal_data)))
print('Outlier Data 개수 : {}'.format(len(outlier_data)))
print(normal_data.describe().T)
print(outlier_data.describe().T)
print(normal_data.isnull().sum())
print(outlier_data.isnull().sum())

# [Code 12, PDF 27] Original normal sensor visualization.
use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
plt.figure(figsize=(30, 15))
for i in range(len(use_col)):
    plt.subplot(3, 1, i + 1)
    plt.title(use_col[i], fontsize=30)
    plt.plot(normal_data[use_col[i]])
    plt.xticks(size=20)
    plt.yticks(size=20)
plt.tight_layout()

# [Code 13, PDF 28] Original anomaly sensor visualization.
use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
plt.figure(figsize=(30, 15))
for i in range(len(use_col)):
    plt.subplot(3, 1, i + 1)
    plt.title(use_col[i], fontsize=30)
    plt.plot(outlier_data[use_col[i]])
    plt.xticks(size=20)
    plt.yticks(size=20)
plt.tight_layout()

# [Code 14, PDF 28] Exact original abs() application; no other preprocessing.
normal_data[use_col] = normal_data[use_col].applymap(lambda x: abs(x))
outlier_data[use_col] = outlier_data[use_col].applymap(lambda x: abs(x))

# [Code 15, PDF 29] Original postprocessing normal visualization.
use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
plt.figure(figsize=(30, 15))
for i in range(len(use_col)):
    plt.subplot(3, 1, i + 1)
    plt.title(use_col[i], fontsize=30)
    plt.plot(normal_data[use_col[i]])
    plt.xticks(size=20)
    plt.yticks(size=20)
plt.tight_layout()

# [Code 16, PDF 30] Original postprocessing anomaly visualization.
use_col = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
plt.figure(figsize=(30, 15))
for i in range(len(use_col)):
    plt.subplot(3, 1, i + 1)
    plt.title(use_col[i], fontsize=30)
    plt.plot(outlier_data[use_col[i]])
    plt.xticks(size=20)
    plt.yticks(size=20)
plt.tight_layout()

# [Codes 17-18, PDF 30-31] Correlations for inspection only.
print(normal_data[use_col].corr())
print(outlier_data[use_col].corr())

# [Code 19, PDF 31] Original row slices; no sort or shuffled split.
X_normal = normal_data[use_col]
y_normal = normal_data['Equipment_state']
X_anomaly = outlier_data[use_col]
y_anomaly = outlier_data['Equipment_state']
X_train_normal = X_normal[:15000]
y_train_normal = y_normal[:15000]
X_test_normal = X_normal[15000:]
y_test_normal = y_normal[15000:]
X_test_anomaly = X_anomaly
y_test_anomaly = y_anomaly

# [Code 20, PDF 32] Fit MinMaxScaler exclusively on normal training rows.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)
y_train_normal = np.array(y_train_normal)
y_test_normal = np.array(y_test_normal)
y_test_anomaly = np.array(y_test_anomaly)

# [Code 21, PDF 33] Original three sliding-window loops and future labels.
sequence = 20
X_train, Y_train = [], []
for index in range(len(X_train_scaled) - sequence - 100):
    X_train.append(X_train_scaled[index: index + sequence])
    Y_train.append(y_train_normal[index + sequence + 100])
X_test_normal, Y_test_normal = [], []
for index in range(len(X_test_normal_scaled) - sequence - 100):
    X_test_normal.append(X_test_normal_scaled[index: index + sequence])
    Y_test_normal.append(y_test_normal[index + sequence + 100])
X_test_anomal, Y_test_anomal = [], []
for index in range(len(X_test_anomaly_scaled) - sequence - 100):
    X_test_anomal.append(X_test_anomaly_scaled[index: index + sequence])
    Y_test_anomal.append(y_test_anomaly[index + sequence + 100])
X_test_normal, Y_test_normal = np.array(X_test_normal), np.array(Y_test_normal)
X_test_anomal, Y_test_anomal = np.array(X_test_anomal), np.array(Y_test_anomal)

# [Code 22, PDF 34] Original validation/final-test slices and stacking order.
X_valid_normal, Y_valid_normal = X_test_normal[:880, :, :], Y_test_normal[:880]
X_test_normal, Y_test_normal = X_test_normal[880:, :, :], Y_test_normal[880:]
X_valid_anomal, Y_valid_anomal = X_test_anomal[:300, :, :], Y_test_anomal[:300]
X_test_anomal, Y_test_anomal = X_test_anomal[300:, :, :], Y_test_anomal[300:]
X_valid = np.vstack((X_valid_normal, X_valid_anomal))
Y_valid = np.hstack((Y_valid_normal, Y_valid_anomal))
X_test = np.vstack((X_test_normal, X_test_anomal))
Y_test = np.hstack((Y_test_normal, Y_test_anomal))

# [Code 23, PDF 34] Preserve conversion timing and print original shapes.
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
X_test, Y_test = np.array(X_test), np.array(Y_test)
print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_valid:', X_valid.shape, 'Y_valid:', Y_valid.shape)
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)

# [Code 24, PDF 35] Only normal validation sequences monitor training.
X_valid_0 = X_valid[Y_valid == 0]
print('X_valid_0:', X_valid_0.shape)
_shapes = {key: list(value.shape) for key, value in
           [('X_train', X_train), ('X_valid', X_valid), ('X_test', X_test), ('X_valid_0', X_valid_0)]}

# [Code 25, PDF 35] Original Sequential/add architecture and input_shape.
def LSTM_AE(sequence, n_features):
    lstm_ae = models.Sequential()
    # Encoder
    lstm_ae.add(layers.LSTM(64, input_shape=(sequence, n_features), return_sequences=True))
    lstm_ae.add(layers.LSTM(32, return_sequences=False))
    lstm_ae.add(layers.RepeatVector(sequence))
    # Decoder
    lstm_ae.add(layers.LSTM(32, return_sequences=True))
    lstm_ae.add(layers.LSTM(64, return_sequences=True))
    lstm_ae.add(layers.TimeDistributed(layers.Dense(n_features)))
    return lstm_ae

lstm_ae = LSTM_AE(20, 3)
lstm_ae.summary()
_initial_weights_sha256 = hashlib.sha256(b''.join(w.tobytes() for w in lstm_ae.get_weights())).hexdigest()
print('Initial weights SHA256:', _initial_weights_sha256)

# [Code 26, PDF 36] Original callbacks, optimizer and fit call defaults.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=50, verbose=1)
es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=120,
                   verbose=1, mode='min', restore_best_weights=True)
lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(0.001))
history = lstm_ae.fit(X_train, X_train,
                     epochs=800, batch_size=128,
                     callbacks=[reduce_lr, es], validation_data=(X_valid_0, X_valid_0))
# Save completed training even if the original threshold rule later fails.
pd.DataFrame(history.history).to_csv(OUTPUT_DIR / 'training_history.csv', index=False)

# [Code 27, PDF 37] Original training-loss plot.
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('loss')
plt.show()

# [Code 28, PDF 37] Preserve np.empty and the original final-timestep loop.
def flatten(X):
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i, X.shape[1] - 1, :]
    return(flattened)

# [Code 29, PDF 38] Exact equality, first matching index, no fallback.
valid_x_predictions = lstm_ae.predict(X_valid)
mse = np.mean(np.power(flatten(X_valid) - flatten(valid_x_predictions), 2), axis=1)
_validation_mse = mse.copy()
precision, recall, threshold = metrics.precision_recall_curve(list(Y_valid), mse)
try:
    index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision, recall)) if p == r][0]
    threshold_final = threshold[index_cnt]
except IndexError:
    # Report original-rule failure; never replace equality with nearest match.
    print('ORIGINAL THRESHOLD RULE FAILED: no usable first precision == recall index.')
    traceback.print_exc()
    (OUTPUT_DIR / 'metrics.json').write_text(json.dumps({
        'status': 'original_threshold_rule_failed', 'shapes': _shapes,
        'environment': _environment, 'baseline_sha256': _baseline_sha256,
        'input_sha256': _inputs_sha256, 'threshold_final': None,
        'reason': 'Original exact-equality selection raised IndexError; compatibility must be a separate file.'
    }, indent=2), encoding='utf-8')
    pd.DataFrame({'split': 'validation', 'label': Y_valid, 'mse': mse}).to_csv(
        OUTPUT_DIR / 'reconstruction_error.csv', index=False)
    raise
plt.figure(figsize=(10, 7))
plt.title('Precision/Recall Curve for threshold', fontsize=15)
plt.plot(threshold[threshold <= 0.2], precision[1:][threshold <= 0.2], label='Precision')
plt.plot(threshold[threshold <= 0.2], recall[1:][threshold <= 0.2], label='Recall')
plt.plot(threshold_final, precision[index_cnt], 'o', color='r', label='Optimal threshold')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()
print('precision: ', precision[index_cnt], ', recall: ', recall[index_cnt])
print('threshold: ', threshold_final)

# [Code 30, PDF 39] Original test error and threshold plot expressions.
test_x_predictions = lstm_ae.predict(X_test)
mse = np.mean(np.power(flatten(X_test) - flatten(test_x_predictions), 2), axis=1)
plt.figure(figsize=(10, 7))
plt.title('Reconstruction Error for both classes', fontsize=15)
plt.plot(np.where(Y_test == 0)[0], mse[Y_test == 0], marker='o', linestyle='', label='Normal')
plt.plot(np.where(Y_test == 1)[0], mse[Y_test == 1], marker='o', linestyle='', label='Anomaly')
plt.axhline(threshold_final, 0, len(Y_test), color='r', linestyle='--', label='Threshold for Anomaly')
plt.legend()
plt.ylabel('Reconstruction Error')
plt.show()

# [Code 31, PDF 40] Original strict > decision, confusion matrix and plot.
pred_y = [1 if e > threshold_final else 0 for e in mse]
conf_matrix = metrics.confusion_matrix(list(Y_test), pred_y)
print('Confusion Matrix:', conf_matrix)
plt.figure(figsize=(7, 7))
sns.heatmap(conf_matrix, xticklabels=[0, 1], yticklabels=[0, 1], annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# [Codes 32-33, PDF 41] Original evaluation; do not tune to reported scores.
print('Accuracy : {}'.format(metrics.accuracy_score(list(Y_test), pred_y)))
print('F1-Score : {}'.format(metrics.f1_score(list(Y_test), pred_y)))

# Required artifacts and clearly separated additional evaluation metrics.
_pr_precision, _pr_recall, _ = metrics.precision_recall_curve(list(Y_test), mse)
_additional = {'precision': float(metrics.precision_score(list(Y_test), pred_y)),
               'recall': float(metrics.recall_score(list(Y_test), pred_y)),
               'balanced_accuracy': float(metrics.balanced_accuracy_score(list(Y_test), pred_y)),
               'roc_auc': float(metrics.roc_auc_score(list(Y_test), mse)),
               'pr_auc_trapezoidal': float(metrics.auc(_pr_recall, _pr_precision))}
print('Additional evaluation metrics:', json.dumps(_additional))
_result = {'status': 'completed', 'baseline': 'Baseline Original', 'shapes': _shapes,
           'threshold_final': float(threshold_final), 'threshold_index': int(index_cnt),
           'threshold_validation_precision': float(precision[index_cnt]),
           'threshold_validation_recall': float(recall[index_cnt]),
           'confusion_matrix': conf_matrix.tolist(),
           'accuracy': float(metrics.accuracy_score(list(Y_test), pred_y)),
           'f1_score': float(metrics.f1_score(list(Y_test), pred_y)),
           'additional_evaluation_metrics': _additional, 'epochs_run': len(history.history['loss']),
           'restored_weights_epoch': int(es.best_epoch + 1), 'environment': _environment,
           'baseline_sha256': _baseline_sha256, 'input_sha256': _inputs_sha256,
           'initial_weights_sha256': _initial_weights_sha256,
           'guidebook_reported': {'accuracy': 0.9751196172248804, 'f1_score': 0.7475728155339806,
                                  'confusion_matrix': [[3922, 78], [26, 154]]}}
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(_result, indent=2), encoding='utf-8')
pd.DataFrame(conf_matrix, index=['actual_normal', 'actual_anomaly'],
             columns=['predicted_normal', 'predicted_anomaly']).to_csv(OUTPUT_DIR / 'confusion_matrix.csv')
pd.concat([
    pd.DataFrame({'split': 'validation', 'label': Y_valid, 'mse': _validation_mse,
                  'prediction': [1 if e > threshold_final else 0 for e in _validation_mse]}),
    pd.DataFrame({'split': 'test', 'label': Y_test, 'mse': mse, 'prediction': pred_y})
], ignore_index=True).to_csv(OUTPUT_DIR / 'reconstruction_error.csv', index=False)
print('BASELINE ORIGINAL COMPLETE; source SHA256:', _baseline_sha256)
