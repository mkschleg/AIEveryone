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
    return jnp.mean(jnp.square(targets-z))


def accuracy(params, batch, predict):
    inputs, targets = batch
    y = targets
    yhat = predict(params, inputs)
    return jnp.mean(jnp.square(y-yhat))

def train_network(layer_size=8,
                  step_size=0.0005,
                  num_epochs=180,
                  batch_size=128):
    rng = random.PRNGKey(0)

    init_random_params, predict = stax.serial(
        Dense(layer_size), Relu,
        Dense(layer_size), Relu,
        Dense(1))

    momentum_mass = 0.9

    train_inputs, train_labels, neg_inputs, neg_labels = datasets.addition()
    num_train = train_inputs.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_inputs[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(
        step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch, predict), opt_state)

    _, init_params = init_random_params(rng, (-1, 2))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    # print(init_params)
    train_acc_arr = []
    test_acc_arr = []
    # print("\nStarting training...")
    for epoch in range(num_epochs):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        params = get_params(opt_state)
        train_acc = accuracy(params, (train_inputs, train_labels), predict)
        test_acc = accuracy(params, (neg_inputs, neg_labels), predict)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    return train_acc_arr, test_acc_arr, (predict, get_params(opt_state))


def get_prediction(predict, params, input_data, normalize=True):
    if normalize:
        input_data = input_data/10.0
    return predict(params, input_data)*20.0


def main():
    train_acc_arr, params = train_network()

if __name__ == "__main__":
    main()
