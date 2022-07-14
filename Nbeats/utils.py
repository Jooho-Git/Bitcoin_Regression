import numpy as np
import string
import random

# scaler
class MeanScaler:
    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean
    
    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


# metric
def MAPEval(y_pred, y_true):
    y_true = np.array(y_true).ravel() + 1e-4
    y_pred = np.array(y_pred).ravel()    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MASEval(x_test, y_test, y_pred):
    x_test = x_test.ravel()
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()

    n = x_test.shape[0]
    d = np.abs(np.diff(x_test)).sum() / (n-1)

    errors = np.abs(y_test - y_pred)
    return errors.mean() / d
    
def SMAPEval(y_pred, y_true):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    return np.mean((np.abs(y_true-y_pred))/(np.abs(y_true) + np.abs(y_pred))) * 100

def MAEval(y_pred, y_true):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    return np.mean(np.abs(y_true-y_pred))    



# generate serial number for version save 
def generate_serial_number():
    string_pool = string.ascii_lowercase + string.digits
    result = ''
    for i in range(3):
        result += random.choice(string_pool)
    return result