import logging
from abc import abstractmethod

# import matplotlib.pyplot as plt
import numpy as np


class BaseMF:

    def __init__(self, epochs, train_data, test_data):
        # rmse
        self.train_error_rmse_ls = []
        self.test_error_rmse_ls = []
        # ame
        self.train_error_mae_ls = []
        self.test_error_mae_ls = []

        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs

    pass

    def store_errors(self, train_error, test_error):
        self.train_error_rmse_ls.append(train_error[0])
        self.test_error_rmse_ls.append(test_error[0])
        self.train_error_mae_ls.append(train_error[1])
        self.test_error_mae_ls.append(test_error[1])

    @abstractmethod
    def sgd(self):
        pass

    @abstractmethod
    def predict_data(self, d):
        pass

    def error(self, data):
        mse = 0
        mae = 0
        total = 0
        error = 0
        for d in data:
            if len(d) > 2:
                try:
                    diff = self.predict_data(d)
                    mse += np.power(diff, 2)
                    mae += abs(diff)
                    total += 1
                except Exception as e:
                    # logging.error(e)
                    error += 1
                    pass
        # logging.info("Error:{}".format(error))
        if total == 0:
            return 0, 0
        return np.sqrt(mse / total), (mae / total)

    def train(self, interval):
        for i in range(self.epochs):
            self.sgd()
            train_e = self.error(self.train_data)
            test_e = self.error(self.test_data)
            self.store_errors(train_e, test_e)
            if i % interval == 0:
                logging.info(
                    "Epochs:{}, Error:{}".format(i, train_e))
                    # "Epochs:{}, Train_Error:{}, Test_Error:{}".format(i, train_e, test_e))

        train_e = self.error(self.train_data)
        test_e = self.error(self.test_data)
        self.store_errors(train_e, test_e)
        logging.info(
            "Epochs:{}, Error:{}".format(i, train_e))
            # "Train_Error:{}, Test_Error:{}".format(train_e, test_e))
        pass

    def plot(self):
        y1 = self.train_error_rmse_ls
        y2 = self.test_error_rmse_ls
        y3 = self.train_error_mae_ls
        y4 = self.test_error_mae_ls
        X = np.arange(0, len(y1), 1)
        # plt.plot(X, y1, label="RMSE - Train")
        # plt.plot(X, y2, label="RMSE - Test")
        # plt.plot(X, y3, label="MAE - Train")
        # plt.plot(X, y4, label="MAE - Test")
        # plt.legend(loc='upper right')
        # plt.show()
