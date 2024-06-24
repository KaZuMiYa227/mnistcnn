import load_data as dl
import validation as vl
import data_process as dp

def main():
    # data loading
    x_train, t_train, x_test, t_test = dl.data_loading()

    vl.validation(x_train, t_train, x_test, t_test)


if __name__ == "__main__":
    main()
