import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import importlib
from sklearn import metrics

from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint

def log_loss(y, y_pred):
    total_loss = 0
    for j in range(9):
        loss = metrics.log_loss(y[:, j], y_pred[:, j])
        total_loss += loss
    total_loss /= 6.

    return total_loss

class Trainer(object):

    def __init__(self, model_stamp, epoch_num, learning_rate=1e-3,
                 shuffle_inputs=False, verbose_round=40, early_stopping_round=8):
        self.models = []
        self.model_stamp = model_stamp
        self.val_loss = -1
        self.auc = -1
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.eps = 1e-10
        self.verbose_round = verbose_round
        self.early_stopping_round = early_stopping_round
        self.shuffle_inputs = shuffle_inputs

    def train_folds(self, X, y, fold_count, batch_size, get_model_func):
        fold_size = len(X) // fold_count
        models = []
        fold_predictions = []
        score = 0
        total_auc = 0

        for fold_id in range(0, fold_count):
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            if fold_id == fold_count - 1:
                fold_end = len(X)

            train_x = np.concatenate([X[:fold_start], X[fold_end:]])
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x = X[fold_start:fold_end]
            val_y = y[fold_start:fold_end]


            model, bst_val_score, auc, fold_prediction = self._train_model_by_logloss(
                    get_model_func(), batch_size, train_x, train_y, val_x, val_y, fold_id)
            score += bst_val_score
            total_auc += auc
            models.append(model)
            fold_predictions.append(fold_prediction)

        self.models = models
        self.val_loss = score / fold_count
        self.auc = total_auc / fold_count
        return models, self.val_loss, self.auc, fold_predictions

    def _train_model_by_logloss(self, model, batch_size, train_x, train_y, val_x, val_y, fold_id):
        early_stopping = EarlyStopping(monitor='val_loss', patience=6)
        bst_model_path = self.model_stamp + str(fold_id) + '.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        hist = model.fit(train_x, train_y,
                         validation_data=(val_x, val_y),
                         epochs=self.epoch_num, batch_size=batch_size, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint])
        bst_val_score = min(hist.history['val_loss'])
        predictions = model.predict(val_x)
        auc = roc_auc_score(val_y, predictions)
        print("AUC Score", auc)
        return model, bst_val_score, auc, predictions