import data_process as dp
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

def validation(x_train, t_train, x_test, t_test):
    network = dp.TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

    # hyper parameters
    iters_num = 100000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # the number of repetition per one epoch
    iter_per_epoch = max(train_size / batch_size, 1)

    # その他の処理
    
    for i in range(iters_num):
        # mini batch calculation
        batch_mask = cp.random.choice(train_size, batch_size).get()

        # x_train と t_train が cupy 配列である場合、NumPy 配列に変換
        if isinstance(x_train, cp.ndarray):
            x_batch = x_train.get()[batch_mask]
            t_batch = t_train.get()[batch_mask]
        else:
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

        # gradient calculation
        grad = network.gradient(x_batch, t_batch)

        # parameters renewal
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # tracking
        if i % iter_per_epoch == 0:
            if isinstance(x_train, cp.ndarray):
                train_acc = network.accuracy(x_train.get(), t_train.get())
                test_acc = network.accuracy(x_test.get(), t_test.get())
            else:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc |" + str(train_acc) + ", " + str(test_acc))

    return train_acc_list, test_acc_list