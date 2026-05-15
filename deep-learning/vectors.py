import numpy as np
# import matplotlib.pylot as plt 

def f(x,n):
    return x**n
x = 3
h = 1e-8
n = 3
derivative = ((f((x+h),n) - f(x,n))/h)
print(derivative)