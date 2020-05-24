from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

def normalize_features(train_x, val_x, test_x,):
    """
    Inputs:
        train_x: train set for specified features
        val_x: validation set for specified features
        test_x: test set for specified features

    Output
        norm_train_x: normalized train set features
        norm_val_x: normalized val set features
        norm_test_x: normalized test set features
    """
    scaler = StandardScaler()
    scaler = scaler.fit(train_x)
    norm_train_x = scaler.transform(train_x)
    norm_val_x = scaler.transform(val_x)
    norm_test_x = scaler.transform(test_x)

    return [norm_train_x.astype('float32'), norm_val_x.astype('float32'), norm_test_x.astype('float32')]

def normalize_targets(train_y, val_y, test_y, stateList):
    """
    Inputs:
        train_y: train set targets
        val_y: validation set targets
        test_y: test set targets

    Output
        train_y_oneHot: train set targets as a one-hot encoder
        train_y_oneHot: train set targets as a one-hot encoder
        train_y_oneHot: train set targets as a one-hot encoder
    """
    list_targets = [train_y, val_y, test_y]
    for dataset in list_targets:
        for idx, state in enumerate(dataset):
            dataset[idx] = stateList.index(state)

    train_y_oneHot = np_utils.to_categorical(train_y, len(stateList))
    val_y_oneHot = np_utils.to_categorical(val_y, len(stateList))
    test_y_oneHot = np_utils.to_categorical(test_y, len(stateList))

    return [train_y_oneHot, val_y_oneHot, test_y_oneHot]



