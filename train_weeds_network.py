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

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""


import time
import itertools

import numpy.random as npr

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, Sigmoid
import datasets

import tqdm

# Assumes the output of network is logsigmoid.
def loss(params, batch, predict):
    inputs, targets = batch
    z = predict(params, inputs)
    return -jnp.mean(targets * jnp.log(z) + (1-targets) * jnp.log(1-z))



def accuracy(params, batch, predict):
    inputs, targets = batch
    target_class = targets
    predicted_class = predict(params, inputs) > 0.5
    return jnp.mean(predicted_class == target_class)

def train_network(layer_size=256,
                  step_size=0.0005,
                  num_epochs=180,
                  batch_size=128):
    rng = random.PRNGKey(0)

    init_random_params, predict = stax.serial(
        Dense(256), Relu,
        Dense(256), Relu,
        Dense(1), Sigmoid)
    
    step_size = 0.0005
    num_epochs = 180
    batch_size = 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.weeds(clean_labels=True)
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(
        step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch, predict), opt_state)

    _, init_params = init_random_params(rng, (-1, 28 * 28 * 3))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    # print(init_params)
    train_acc_arr = []
    test_acc_arr = []
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels), predict)
        test_acc = accuracy(params, (test_images, test_labels), predict)
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    return train_acc_arr, test_acc_arr, get_params(opt_state)


def main():
    train_acc_arr, test_acc_arr, params = train_network()

    np.save("network_params.npy",
            np.array(params, dtype='object'),
            allow_pickle=True)
    np.savez("weeds_network_accuracy.npz",
             test=np.array(test_acc_arr),
             train=np.array(train_acc_arr))


def run_model(model_file):
    init_random_params, predict = stax.serial(
        Dense(256), Relu,
        Dense(256), Relu,
        Dense(1), Sigmoid)

    params = np.load(model_file, allow_pickle=True)
    train_images, train_labels, test_images, test_labels = datasets.weeds(clean_labels=True)
    train_acc = accuracy(params, (train_images, train_labels), predict)
    test_acc = accuracy(params, (test_images, test_labels), predict)
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")


if __name__ == "__main__":
    main()
