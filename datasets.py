# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""

import array
import gzip
import os
from os import path
import struct
import urllib.request
import glob
from PIL import Image
import numpy as np
import pandas
import tqdm


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()),
                            dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.)
    test_images = _partial_flatten(test_images) / np.float32(255.)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def addition():
    rng = np.random.RandomState(42)

    train_data = rng.rand(10000, 2) * 10
    train_labels = train_data[:, 0] + train_data[:, 1]
    train_labels = train_labels.reshape((10000, 1))

    neg_data = rng.rand(100, 2) * -10
    neg_labels = neg_data[:, 0] + neg_data[:, 1]
    neg_labels = neg_labels.reshape((100, 1))
    return train_data/10, train_labels/20, neg_data/10, neg_labels/20


def weeds_pre_proc(dataset_folder, output_folder):
    """The pre-processing function ran on the original dataset.
    Students will not need to run this function, done pre distribution."""
    image_files = glob.glob(os.path.join(dataset_folder, "raw_images", "*.jpg"))
    rng = np.random.RandomState(42)
    subset_features = rng.randint(256*256, size=28*28)
    labels_df = pandas.read_csv(os.path.join(output_folder, "labels.csv"))
    labels_df.set_index("Filename", inplace=True)
    feat_list = []
    label_list = []
    pbar = tqdm.tqdm(image_files)
    for fp in pbar:
        img_arr = np.array(Image.open(fp))
        # Only do subselection and flatten here,
        # do pre-proc norm during training.
        subset_img = img_arr.reshape(256*256, 3)[subset_features, :].flatten()
        feat_list.append(subset_img)
        label_row = labels_df.loc[os.path.basename(fp)]
        label_list.append(label_row["Label"])

    full_data = np.stack(feat_list)
    labels = np.array(label_list)
    np.save(os.path.join(output_folder, "weeds_features.npy"), full_data)
    np.save(os.path.join(output_folder, "weeds_labels.npy"), labels)
    return


def weeds_load(folder, train_perc=0.8, split_random_seed=None, clean_labels=False):
    feats = np.load(os.path.join(folder, "weeds_features.npy"))
    labels = _one_hot(np.load(os.path.join(folder, "weeds_labels.npy")) == 8,
                      1)
    # labels = _one_hot(np.load(os.path.join(folder, "weeds_labels.npy")), 9)
    a = np.random.RandomState()
    if split_random_seed:
        a.seed(split_random_seed)
    idxs = np.arange(feats.shape[0])
    a.shuffle(idxs)
    train_feats = feats[idxs[0:int(train_perc*len(idxs))], :] / 256.
    train_labels = labels[idxs[0:int(train_perc*len(idxs))]]
    validation_feats = feats[idxs[int(train_perc*len(idxs))+1:], :] / 256.
    validation_labels = labels[idxs[int(train_perc*len(idxs))+1:]]
    return train_feats, train_labels, validation_feats, validation_labels


def weeds(clean_labels=False):
    return weeds_load("./dataset/",
                      split_random_seed=42,
                      clean_labels=clean_labels)
