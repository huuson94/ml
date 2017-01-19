import math
import numpy as np
from random import uniform
def ca_alg():
    #constant
    a = 0.6380631366077803
    b = 0.5959486060529070
    q = 0.9339962957603650
    W = 0.2488702280083841
    A = 0.6366197723675813
    B = 0.5972997593539963
    H = 0.0214949004570452
    P = 4.9125013953033204
    
    U = uniform(0, 1)
    T = U - 0.5
    S = W - T^2
    if(S > 0):
        Z = T * (A / S + B)
        return Z
    while True:
        U = uniform(0, 1)
        U_ = uniform(0, 1)
        T = U - 0.5
        S = 1/4 - T^2
        Z = T * (a / S + b)
        if((S^2 * ((1 + Z^2 )*(H*U_ + P ) - q) + S) > 0.5):
            break

    
def ea_alg():
    #constant
    q = math.log(2)
    a = 5.7133631526454228
    b = 1.4142135623730950
    c = -1.6734053240284925
    p = 0.9802581434685472
    A = 5.6005707569738080
    B = 3.3468106480569850
    H = 0.0026106723602095
    D = 0.0857864376269050

    U = uniform(0, 1)
    K = c
    U = U + U
    while (U < 1):
        U = U + U
        K = K + q
    U = U - 1
    if (U <= p):
        Z = K + A / (B - U)
        return Z
    while True:
        U = uniform(0, 1)
        U_ = uniform(0, 1)
        Y = a / (b - U)
        if( (U_ * H + D) * (b - U)^2 < e ^ (-Y - c)):
            break
    Z = K + Y
    return Z

def na_alg():
    U = uniform(0,1)
    if(U < 1/2):
        B = 0
    else:
        B = 1
    V = ea_alg()
    S = V + V
    W = ca_alg()
    Z = math.sqrt(S / (1 + W^2))
    Y = W*Z
    if B == 0:
        return Z, Y
    else:
        return -Z, Y

def modify_sample(image, number_sample = 10, epsilon = 1):
    add_values = []
    for index in range(number_sample):
        add_values.append(np_alg()[0])
        add_values.append(np_alg()[1])
    mean = np.median(numpy.array(add_values))
    image += mean * epsilon
    return image



