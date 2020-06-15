import numpy as np
import matplotlib.pylab as plt

# numerical diff
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# function
def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0*2.0 + x1*x1

print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)

        x[idx] = temp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = temp_val
    
    return grad

def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    
    return x


init_x = np.array([-3.0, 4.0])
#print("gradient_descsent : ", gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# too less learning rate
#print("gradient_descsent_too less lr : ", gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))

# too much learning rate
print("gradient_descsent_too much lr : ", gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))

