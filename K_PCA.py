import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import pca
stride = 1024
pca = pca.PCA(n_components=1, copy=False, svd_solver="full")
X = sio.loadmat("1.mat")["buffer"]
[t, n] = X.shape
out = np.ones([X.shape[0],])
for i in range(int(t/1024)):
    this_range = np.arange(i*1024, i*1024+1024)
    c = X[this_range]
    out[this_range] = np.squeeze(pca.fit_transform(c))

out = (out - np.min(out))/(np.max(out) - np.min(out))
out = (out - 0.5)*2
mask = np.isnan(out)
out[mask] = 0
mask = np.isinf(out)
out[mask] = 0
out = out.astype(np.float)
wavfile.write(filename="./haah.wav", data=out, rate=45000)