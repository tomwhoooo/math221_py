## Adapted from Professor Per-Olof Persson's Math 221 Homework 2 MatLab Scripts
## Developed under python3, numpy version 1.13
## Tom Hu or Tom Who Hates MatLab @Spring 2018

import numpy as np

def clgs(A):
    m, n = np.shape(A)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:,j]
        for i in range(j - 1):
            R[i, j] = np.matrix.getH(Q[:, i]) * A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return(np.asmatrix(Q), np.asmatrix(R))

#### Test Scripts

#A = np.random.rand(3, 2)
#Q, R = clgs(A)
#np.linalg.norm(Q[:,0])
#R
#Q * R == A

def mgs(A):
    m, n = np.shape(A)
    V = A
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 2, n):
            R[i, j] = np.dot(np.matrix.getH(Q[:,i]), (V[:,j]))
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return(np.asmatrix(Q), np.asmatrix(R))

#### Test Scripts ####

#A = np.random.rand(3, 2)
#Q, R = mgs(A)
#np.linalg.norm(Q[:,0])
#R
#Q * R == A

## Complex numbers have not been tested for the following scripts
## majorly due to numpy has to convert an ndarray to a matrix in order to 
## compute the conjugate transpose. I would expect the behavior might be weird...

def house(A):
    m, n = np.shape(A)
    W = np.zeros((m, n))
    for k in range(n):
        v = A[np.ix_(range(k,m), range(k, k+1))]
        #print(v)
        v[0] = v[0] + (2 * (v[0] >= 0) - 1) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)
        #print(v)
        W[np.ix_(range(k,m), range(k, k+1))] = v
        A[np.ix_(range(k, m), range(k, n))] = A[np.ix_(range(k, m), range(k, n))] - 2 * np.outer(v, (np.dot(np.matrix.getH(v), A[np.ix_(range(k, m), range(k, n))])))
    R = np.triu(A[np.ix_(range(0, n), range(0, n))])
    return(W, R)

def formQ(W):
    m, n = np.shape(W)
    Q = np.identity(m)
    for k in reversed(range(n)):
        Q[np.ix_(range(k, m), range(m))] = Q[np.ix_(range(k, m), range(m))] - 2 * np.dot(W[np.ix_(range(k, m), range(k, k+1))], (np.dot(np.matrix.getH(W[np.ix_(range(k, m), range(k, k+1))]), Q[np.ix_(range(k, m), range(m))])))
    return(Q)


#### Test Scripts ####
## you can add disp(v) in corresponding place in Professor's MatLab scripts
## Note that the . after the number is crucial, you can test without the .
## and see what happened. Very interesting failure!

#A = np.matrix(([1.,3.],[1.,2.],[2.,3.]))
#house(A)
#W, R = house(A)
#Q = formQ(W)
#R
#Q

