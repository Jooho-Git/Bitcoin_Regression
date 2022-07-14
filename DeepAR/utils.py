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
    for i in range(7):
        result += random.choice(string_pool)
    return result