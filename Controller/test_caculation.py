import scipy.io as sio
import numpy as np
import ILRMA

ILRMA = ILRMA.ILRMA()
B = sio.loadmat("./test_mat/haha.mat")
X = B["X"]
T = B["T"]
V = B["V"]
sep, cost, W = ILRMA.ilrmabody(X, T, V)
sio.savemat("out_.mat",{"out": sep})

