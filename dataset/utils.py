import numpy as np
import pandas

def load_raw_data(images_path, labels_path, train_ind_path, val_ind_path, width=56, height=56, n_channels=3):
    # Load images
    with open(images_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    data = data.reshape(-1, height, width, n_channels)
    data = data.transpose(0, 3, 1, 2)

    # Load labels
    df = pandas.read_csv(labels_path, header=None)
    labels = df.to_numpy().T[0]

    # Build train set and validation set
    train_indices = np.load(train_ind_path)
    val_indices = np.load(val_ind_path)
    images_train, labels_train = data[train_indices], labels[train_indices]
    images_test, labels_test = data[val_indices], labels[val_indices]

    return images_train, images_test, labels_train, labels_test

def load_test_data(images_path, width=56, height=56, n_channels=3): 
    # Load images
    with open(images_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    data = data.reshape(-1, height, width, n_channels)
    data = data.transpose(0, 3, 1, 2)

    return data