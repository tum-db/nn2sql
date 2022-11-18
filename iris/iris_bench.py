import numpy as np
import time
import sys

lr=0.2 # learningrate
attss=(20, 50)
iters=(10, 100, 1000)
limits=(150, 300, 600, 1200)


def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))

tmp = np.loadtxt("iris.csv", delimiter=",", dtype=float,skiprows=1)
arr = np.concatenate((tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp))
X = arr[:,0:4]/10
y = arr[:,4].astype(int)
y_oh = np.zeros((y.size, y.max()+1))
y_oh[np.arange(y.size),y] = 1
y=y_oh

print("name, atts, limit, lr, iter, execution_time, error")
for atts in attss:
  np.random.seed(1)
  w_xh = 2*np.random.random((X[0].size,atts)) - 1  # (784*20)
  w_ho = 2*np.random.random((atts,y[0].size)) - 1         # 20*10

  for limit in limits:
    for iter in iters:
        start = time.time()
        for j in range(iter):
            print("Iteration: " + str(j), file=sys.stderr)
            a_xh = nonlin(np.dot(X[0:limit,:],w_xh))
            a_ho = nonlin(np.dot(a_xh,w_ho))
            l_ho = 2*(a_ho - y[0:limit,:])
            error=str(np.mean(np.abs(l_ho)))
            print("Error:" + error, file=sys.stderr)
            d_ho = l_ho*nonlin(a_ho,deriv=True)
            l_xh = d_ho.dot(w_ho.T)
            d_xh = l_xh * nonlin(a_xh,deriv=True)
            w_ho -= lr * a_xh.T.dot(d_ho)
            w_xh -= lr * X[0:limit,:].T.dot(d_xh)
        end = time.time()
        print("numpy,", atts, ",", limit, ",", lr, ",", iter, ",", end-start, ",", error)
