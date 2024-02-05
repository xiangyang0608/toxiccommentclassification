import numpy as np
import pandas as pd

def worst_group_accuracy(prediction, y):
    """
        Compute the worst group accuracy, with the groups being defined by ['male', 'female', 'LGBTQ',
        'christian', 'muslim', 'other_religions', 'black', 'white'] for positive and negative toxicity.
        arguments:
            prediction [pandas.DataFrame]: dataframe with 2 columns (index and pred)
            y [pandas.DataFrame]: dataframe containing the metadata
        returns:
            wga [float]: worst group accuracy
    """
    y.loc[prediction.index, 'pred'] = prediction.pred

    categories = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']
    accuracies = []
    for category in categories:
        for label in [0, 1]:
            group = y.loc[y[category] == label]
            group_accuracy = (group['y'] == (group['pred'] > 0.5)).mean()
            accuracies.append(group_accuracy)
    
    acc_index = 1
    for cate in categories:
        for label in [1]:
            print(f'{cate}_{label}: {accuracies[acc_index]}')
            acc_index += 2
    wga = np.min(accuracies)
    return wga

def save_single_model(models, val_x, test_x, fold_predictions, folds_WGA, model_name):
    fold_score = folds_WGA
    train_predictions = np.concatenate(fold_predictions, axis=0)
    val_predictions = []
    test_predictions = []
    for fold_id, model in enumerate(models):
        val_predicts = model.predict(val_x, batch_size=256, verbose=1)
        val_predictions.append(val_predicts)
        test_predicts = model.predict(test_x, batch_size=256, verbose=1)
        test_predictions.append(test_predicts)
    avg_val_predicts = np.zeros(val_predictions[0].shape)
    for val_pred in val_predictions:
        avg_val_predicts += val_pred
    avg_val_predicts /= len(val_predictions)
    avg_test_predicts = np.zeros(test_predictions[0].shape)
    for test_pred in test_predictions:
        avg_test_predicts += test_pred
    avg_test_predicts /= len(test_predictions)
    
    score = pd.DataFrame(fold_score)
    score.to_csv("metric_"+model_name+".csv", index=False)
    train_pred = pd.DataFrame(train_predictions)
    train_pred.to_csv("train_pred_"+model_name+".csv", index=False)
    val_pred = pd.DataFrame(avg_val_predicts)
    val_pred.to_csv("val_pred_"+model_name+".csv", index=False)
    test_pred = pd.DataFrame(avg_test_predicts)
    test_pred.to_csv("test_pred_"+model_name+".csv", index=False)
    
    return fold_score, train_predictions, avg_val_predicts, avg_test_predicts