"""
copyright from Daichi Kitamura,the org paper is
http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf
Author Daichi Kitamura1
And reference code is from https://github.com/d-kitamura/ILRMA/tree/master/ILRMA_release20180411
for python version is made by yujiacheng333 bupt
concat me: yujiacheng333@bupt.edu.cn
the params shown in paper(which is the default params settings in the program):
Sampling frequency 16 kHz FFT length 128 ms
Window shift length 64 ms Number of bases 10 bases for each speech source
and 4 bases for each music source Initialization of Mixing matrices estimated by S
oftmixing matrices LOST [41] and permutation solver [15]
Initialization of Pretrained bases and activations source models using simple NMF based on
Kullback(NMF variables) Leibler divergence with sources estimated by Soft-LOST and [15]
Annealing for Annealing with noise EM algorithm injection proposed in [27]
Number of iterations 500
"""
import numpy as np
import OPS
import json_reader
from sklearn.decomposition import pca
import matplotlib.pyplot as plt
import scipy.io as sio
np.set_printoptions(precision=4)
class ILRMA(object):

    """
        the implementation of ILRMA1 in org paper
    """
    def __init__(self, path="../data.json"):
        np.random.seed(200)
        self.params = json_reader.JsonReader(path=path).load_json()
        self.Iter = self.params["iterate"]
        if self.params["base_num"]:
            self.n_base = self.params["base_num"]
        else:
            self.n_base = None
        self.n_source = self.params["n_source"]
        self.normalize = self.params["normalize"]
        self.drawConv = self.params["drawConv"]
        self.ref_mic = self.params["ref_mic"]

    def bss_ILRMA(self, mix_audio):
        """
        the main method of ILRMA
        :param mix_audio: the mix audio to be separate
        :return: the separated audios shape = [t, n_source]
        """
        ops = OPS.OPS()
        res = []
        mix_audio = mix_audio.T if mix_audio.shape[0] > mix_audio.shape[1] else mix_audio  # make sure dim 1 is t
        for val in mix_audio:
            res.append(ops.stft(val)[2])
        res = np.asarray(res, dtype=np.complex)
        res = np.transpose(res, [1, 2, 0])
        res_white = self.whitting(res, model=1)
        [y, cost, W] = self.ilrmabody(X=res_white)
        Z = self.backprojection(Y=y, X=res[:, :, self.ref_mic])
        sep = []
        Z = np.transpose(Z, [2, 0, 1])
        for z in Z:
            sep.append(ops.istft(z))
        sep = np.asarray(sep)
        return sep, cost

    def ilrmabody(self, X, T=[], V=[]):
        """
        the steps of ILRMA
        :param X:  the STFT mat with N_channels
        :param T: init params of T default None
        :param V: init params of V default None
        :return: W: initial demixing matrix (source x channel x frequency bin, default: identity matrices)
                 T: initial basis matrix (frequency bin x basis x source in ILRMA1)
                 V: initial activation matrix (basis x time frame x source in ILRMA1)
        """
        if len(np.shape(X)) != 3:
            raise ValueError("Wrong dim of input X:the STFT mat with N_channels")
        [f, t, n_channels] = np.shape(X)
        if not self.n_base:
            buffer = int(t / 10)
            if (buffer - t) > 0.5:
                self.n_base = buffer + 1
            else:
                self.n_base = buffer
        if n_channels > t:
            raise ValueError('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).')
        W = np.zeros([self.n_source, n_channels, f], dtype=np.float32)
        if W.shape[0] != W.shape[1]:
            W = self.valueprotect(np.random.normal(size=W.shape))
            W = W + self.valueprotect(np.random.normal(size=W.shape))*1j
        else:
            for i in range(f):
                W[:, :, i] = np.eye(W.shape[0])
        W = W.astype(np.complex)
        if len(T) == 0 and len(V) == 0:
            T = np.random.uniform(low=0, high=1, size=[f, self.n_base, self.n_source])
            V = np.random.uniform(low=0, high=1, size=[self.n_base, t, self.n_source])
        T = self.valueprotect(T)
        V = self.valueprotect(V)
        R = np.zeros([f, t, self.n_source])
        Y = R.astype(np.complex)
        for i in range(f):
            Y[i, :, :] = np.transpose(np.matmul(W[:, :, i],
                                                np.squeeze(X[i, :, :]).T.astype(np.complex)))
            # shape is [f, n_channel, n_source]
        P = np.power(np.abs(Y), 2)
        E = np.eye(self.n_source)
        Xp = np.transpose(X, [2, 1, 0])
        for n in range(self.n_source):
            R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
        UN1 = np.ones([self.n_source, 1])  # unit vector for expand the single value shape = [self.n_source,1]
        U1J = np.ones([1, t])  # unit vector for expand the single value shape = [1, t]
        cost = np.zeros([self.Iter, 1])
        cost[0, 0] = self.cost_function_local(P, R, W, f, t)
        for it in range(self.Iter):
            print("Iteration at {}".format(it))
            for n in range(self.n_source):
                # update T
                buffer = (1 / R[:, :, n])
                T[:, :, n] = T[:, :, n] * np.sqrt(np.matmul(P[:, :, n] / np.power(R[:, :, n], 2),
                                                            V[:, :, n].T) / np.matmul(1 / R[:, :, n], V[:, :, n].T))
                T[:, :, n] = self.valueprotect(T[:, :, n])
                R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
                # update V
                V[:, :, n] = V[:, :, n] * np.sqrt(np.matmul(T[:, :, n].T, (P[:, :, n] / np.power(R[:, :, n], 2))) /
                                                  np.matmul(T[:, :, n].T, 1 / R[:, :, n]))
                V[:, :, n] = self.valueprotect(V[:, :, n])
                R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
                for i in range(f):
                    # U = 1 / t * np.matmul(X[i, :, :].T,
                    #                      X[i, :, :] * np.matmul(np.expand_dims(1 / R[i, :, n], -1), UN1.T)).T

                    Q = Xp[:, :, i] * np.matmul(UN1, U1J / R[i, :, n])
                    U = np.matmul(Q, Xp[:, :, i].conj().T)/t
                    # U is the M, M mat M is the length = n_channel
                    v = np.matmul(W[:, :, i], U)
                    #w = np.matmul(np.linalg.inv(v), E[:, n])
                    w = np.matmul(np.linalg.solve(v, np.eye(v.shape[0])), E[:, n])
                    w /= np.sqrt(np.matmul(np.matmul(w.conj().T, U), w))
                    W[n, :, i] = w.conj().T
                    buffer = np.matmul(w.conj().T, Xp[:, :, i])
                    Y[i, :, n] = np.matmul(w.conj().T, Xp[:, :, i])

            P = np.power(np.abs(Y), 2)
            P = self.valueprotect(P)

            if self.normalize:
                zeta = np.sqrt(np.sum(np.sum(P, axis=0), axis=0) / (t * f))
                d = self.repmat(zeta, [n_channels, f])
                W /= d
                zetaIJ = np.power(self.repmat(zeta, [f, t]), 2)
                zetaIJ = np.transpose(zetaIJ, [1, 2, 0])
                P /= zetaIJ
                R /= zetaIJ
                zetaIL = np.power(self.repmat(zeta, [f, self.n_base]), 2)
                zetaIL = np.transpose(zetaIL, [1, 2, 0])
                T /= zetaIL
            cost[it, 0] = self.cost_function_local(P, R, W, f, t)
        if self.drawConv:
            plt.plot(cost[:, 0])
            plt.show()
        return Y, cost, W

    def loacl_inv(self, X):
        if len(X.shape) !=2:
            raise ValueError("dim should be 2")
        else:
            [S, V, D] = np.linalg.svd(X)
            V = self.valueprotect(V)
            V = 1 / V
            V = np.diag(V)
            return np.matmul(np.matmul(S, V), D)

    @staticmethod
    def cost_function_local(P, R, W, f, t):
        """
        caculate loss of LIRMA at the org code line 300
        :param P: the Pmat
        :param R: the Rmat
        :param W: the Wmat
        :param f: the f vector as frequency
        :param t: the time vector
        :return: the cost at this epoch
        """
        eps = 2.2204e-16
        A = np.zeros([f, 1])
        for i in range(f):
            x = np.abs(np.linalg.det(W[:, :, i]))
            if x == 0:
                x = eps
            A[i] = np.log(x)
        return np.sum(np.sum(P / R + np.log(R), axis=2)) - 2 * t * np.sum(A)

    @staticmethod
    def valueprotect(X):
        """

        :param X: arbitrary mat to avoid underflow
        :return: the fixed mat
        """
        eps = 2.2204e-16
        if X.dtype == "complex128":
            if np.abs(X).min() < eps:
                X += eps
        else:
            mask = X < eps
            X[mask] = eps
            return X

    @staticmethod
    def repmat(X, out_shape):
        """
        expand vector in to rep mat
        :param X: 1 dim vector
        :param out_shape: shape to be expand
        :return: ~~
        """
        if len(X.shape) == 1 and len(out_shape) == 2:
            re = np.zeros([len(X), out_shape[0], out_shape[1]])
            for i, mat in enumerate(re):
                re[i, :, :] = X[i]
            return re
        else:
            raise ValueError("only for 1 dim vector")

    @staticmethod
    def backprojection(Y, X):
        """
        passed test
         D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, "Determined
         blind source separation with independent low-rank matrix analysis,"
         Audio Source Separation. Signals and Communication Technology.,
         S. Makino, Ed. Springer, Cham, pp. 125-155, March 2018.

         See also:
         http://d-kitamura.net
         http://d-kitamura.net/en/demo_rank1_en.htm
        :param Y: estimated (separated) signals (frequency bin x time frame x source)
        :param X:reference channel of observed (mixture) signal (frequency bin x time frame x 1)
       or observed multichannel signals (frequency bin x time frame x channels)
        :return:scale-fitted estimated signals (frequency bin x time frame x source)
       or scale-fitted estimated source images (frequency bin x time frame x source x channel)
        """
        [I, J, M] = Y.shape
        if len(X.shape) == 2:
            X = np.expand_dims(X, -1)
            A = np.zeros([1, M, I]).astype(np.complex)
            Z = np.zeros([I, J, M]).astype(np.complex)
            for i in range(I):
                Yi = Y[i, :, :].squeeze().T
                A[0, :, i] = np.matmul(np.matmul(X[i, :, 0], Yi.T), np.linalg.inv(np.matmul(Yi, Yi.T)))

            mask = np.isnan(A)
            A[mask] = 0
            mask = np.isinf(A)
            A[mask] = 0
            for m in range(M):
                for i in range(I):
                    Z[i, :, m] = A[0, m, i] * Y[i, :, m]

        else:
            raise ValueError("The number of channels in X must be 1 or equal to that in Y.")
        return Z

    def whitting(self, X, model=0):
        """
        almost same as it in org code use PCA an whitting
        use sklearn PCA method to reduce the dim which bigger than input channel_num
        :param X: the recv mat of audio shape = [f, t, n_channels]
        :param model: the option for pca or turely whitting
        :return: The num of channels in the josn data which shape = [f, t, num_source]
        """
        if not model:
            local_pca = pca.PCA(n_components=self.n_source, svd_solver="full", whiten=True)
            res = []
            angle = np.angle(X)
            X = np.abs(X)
            for i in range(X.shape[1]):
                res.append(local_pca.fit_transform(X[:, i, :]))
            res = np.asarray(res).astype(np.complex)
            res = np.transpose(res, [1, 0, 2])
            res *= np.exp(1j * angle)
            return res
        elif model == 1:
            dnum = self.n_source
            [I, J, M] = X.shape
            Y = np.zeros([I, J, self.n_source], dtype=np.complex)
            for i in range(I):
                Xi = np.squeeze(X[i, :, :]).T
                V = np.matmul(Xi, Xi.T.conj()) / J
                [D, P] = np.linalg.eig(V)
                idx = np.argsort(D)
                D = D[idx]
                P = P[:, idx]
                D = np.diag(D)
                D2 = D[M - dnum:M, M-dnum:M]
                P2 = P[:, M-dnum:M]
                D2_ = np.linalg.pinv(np.sqrt(D2))
                Y[i, :, :] = np.matmul(np.matmul(D2_, P2.conj().T), Xi).T
            return Y
        else:
            raise ValueError("model should in 0 1")

    @staticmethod
    def pop_sort(d):
        """
        simple pop sort for arr
        :param d: input arr
        :return: arg sorted arr and the idx
        """
        idx = np.arange(len(d))
        for i in range(len(d)):
            for j in range(len(d) - i - 1):
                if d[j + 1] < d[j]:
                    buffer = d[j]
                    d[j] = d[j+1]
                    d[j+1] = buffer
                    buffer = idx[j]
                    idx[j] = idx[j+1]
                    idx[j+1] = buffer
        return d, idx

    @staticmethod
    def get_idx(d, idx):
        """

        :param d: org 1dmin array or mat
        :param idx: the rank of arr
        :return: the ranked arr
        """
        if len(d) != len(idx):
            raise ValueError("Need to be same shape")
        else:
            res = []
            for i in range(len(idx)):
                res.append(d[idx[i]])
            return np.asarray(res, dtype=np.complex)


if __name__ == '__main__':
    X = [[1,2,3,4],[2,3,4,5],[7,7,7,7]]
    X = np.asmatrix(X, dtype=np.complex)
    X = [X, X, X]
    X = np.asarray(X)
    X = np.transpose(X, [1, 2, 0])
    A = ILRMA().whitting(X, model=1)
