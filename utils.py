import os
import tarfile
import urllib
import numpy as np
import pandas as pd

from zlib import crc32

def fetch_data(url, path):
    # retrieve housing prices data
    os.makedirs(path, exist_ok=True)
    data_file_name = os.path.basename(path)
    if (os.path.exists(path)):
        return
    tgz_path = os.path.join(path, f"{data_file_name}.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=path)
    housing_tgz.close()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    # some 32 bit integer hashed  is less than test_ratio * 2**32
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data: pd.DataFrame, test_ratio: float, id_column: str):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id: test_set_check(id, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
