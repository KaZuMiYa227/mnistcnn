import load_data as dl
import validation as vl
import data_process as dp

import os
import time
import h5py

def main():
    # create unique save directory
    save_dir = os.path.join("./results", time.strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # data loading
    x_train, t_train, x_test, t_test = dl.data_loading()

    train_acc_list, test_acc_list, train_loss_list, test_loss_list = vl.validation(x_train, t_train, x_test, t_test)

    # save the result
    with h5py.File(os.path.join(save_dir, "result.h5"), "w") as f:
        f.create_dataset("train_acc_list", data=train_acc_list)
        f.create_dataset("test_acc_list", data=test_acc_list)
        f.create_dataset("train_loss_list", data=train_loss_list)
        f.create_dataset("test_loss_list", data=test_loss_list)

if __name__ == "__main__":
    main()
