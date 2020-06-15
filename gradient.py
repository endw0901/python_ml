import numpy as np
import matplotlib.pylab as plt

# # numerical diff
# def numerical_diff(f,x):
#     h = 1e-4
#     return (f(x+h) - f(x-h)) / (2*h)

# # function
# def function_2(x):
#     return x[0]**2 + x[1]**2

# def function_tmp1(x0):
#     return x0*x0 + 4.0**2.0

# def function_tmp2(x1):
#     return 3.0*2.0 + x1*x1

# print(numerical_diff(function_tmp1, 3.0))
# print(numerical_diff(function_tmp2, 4.0))

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))