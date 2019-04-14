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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ILRMA(object):

    """
        the implementation of ILRMA1 in org paper
    """
    def __init__(self, path="../data.json"):
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
            res.append(ops.stft(val))
        res = np.asarray(res)
        # print(res.shape)
        res = np.transpose(res, [1, 2, 0])
        res_white = self.whitting(res)
        # print(res_white.shape)
        [y, _, _] = self.ilrmabody(X=res_white)
        Z = self.backprojection(Y=y, X=res[:, :, self.ref_mic])
        sep = []
        Z = Z.transpose(Z, [2, 0, 1])
        for z in Z:
            sep.append(ops.istft(z))
        sep = np.asarray(sep)
        return sep

    def ilrmabody(self, X, T = None, V = None ):
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
            for i in range(W.shape[2]):
                W[:, :, i] = np.eye(W.shape[0])
                W[:, :, i] = self.valueprotect(W[:, :, i])
        if not (T and V):
            T = np.random.normal(size=[f, self.n_base, self.n_source])
            V = np.random.normal(size=[self.n_base, t, self.n_source])
        T = self.valueprotect(T)
        V = self.valueprotect(V)
        R = np.zeros([f, t, self.n_source])
        Y = R
        for i in range(f):
            Y[i, :, :] = np.transpose(np.matmul(W[:, :, i], np.transpose(X[i, :, :].squeeze())))
            # shape is [f, n_channel, n_source]
        P = np.power(np.abs(Y), 2)
        E = np.eye(self.n_source)
        Xp = np.transpose(X, [2, 1, 0])
        for n in range(self.n_source):
            R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
        UN1 = np.ones([self.n_source, 1])  # unit vector for expand the single value shape = [self.n_source,1]
        U1J = np.ones([1, t])  # unit vector for expand the single value shape = [1, t]
        cost = np.zeros([self.Iter + 1, 1])
        for it in range(self.Iter):
            cost[0, 0] = self.cost_function_local(P, R, W, f, t)
            for n in range(self.n_source):
                # update T
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
                    U = (np.matmul(Xp[:, :, i] * np.matmul(UN1, (U1J / R[i, :, n])), Xp[:, :, i].T))/t
                    # U is the M, M mat M is the length = n_channel
                    w = np.matmul(np.linalg.pinv(np.matmul(W[:, :, i], U)), E[:, n])
                    w /= np.matmul(np.matmul(np.sqrt(w.T), U), w)
                    W[n, :, i] = w.T
                    Y[i, :, n] = np.matmul(w.T, Xp[:, :, i])
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
        A = np.zeros([f, 1])
        for i in range(f):
            x = np.abs(np.linalg.det(W[:, :, i]))
            if x == 0:
                x = 1e-12
            A[i] = np.log(x)
        return np.sum(np.sum(P/R + np.log(R + 1e-12), axis=2)) - 2 * t * np.sum(A)

    @staticmethod
    def valueprotect(X):
        """

        :param X: arbitrary mat to avoid underflow
        :return: the fixed mat
        """
        mask = X < 1e-12
        X[mask] = 1e-12
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
        [I, J, M] = Y.shape()
        if X.shape[2] == 1:
            A = np.zeros(1, M, I)
            Z = np.zeros(I, J, M)
            for i in range(I):
                Yi = Y[i, :, :].squeeze().T
                A[0, :, i] = np.matmul(X[i, :, 0], Yi.T) / np.matmul(Yi, Yi.T)
            mask = np.isnan(A)
            A[mask] = 0
            mask = np.isinf(A)
            A[mask] = 0
            for m in range(M):
                for i in range(I):
                    Z[i, :, m] = np.matmul(A[0, :, i], Y[i, :, m])
        elif X.shape[2] == M:
            A = np.zeros(M, M, I)
            Z = np.zeros(I, J, M, M)
            for i in range(I):
                for m in range(M):
                    Yi = Y[i, :, :].squeeze().T
                    A[m, :, i] = np.matmul(X[i, :, m], Yi.T) / np.matmul(Yi, Yi.T)
            mask = np.isnan(A)
            A[mask] = 0
            mask = np.isinf(A)
            A[mask] = 0
            for n in range(M):
                for m in range(M):
                    for i in range(I):
                        Z[i, :, n, m] = A[m, n, i] * Y[i, :, n]
        else:
            raise ValueError("The number of channels in X must be 1 or equal to that in Y.")
        return Z

    def whitting(self, X):
        """
        use sklearn PCA method to reduce the dim which bigger than input channel_num
        :param X: the recv mat of audio shape = [f, t, n_channels]
        :return: The num of channels in the josn data which shape = [f, t, num_source]
        """
        D_num = self.n_source
        pca = PCA(n_components=D_num, whiten=True)
        F = X.shape[0]
        res = np.zeros([X.shape[0], X.shape[1], D_num])
        for f in range(F):
            res[f, :, :] = pca.fit_transform(X[f, :, :])
        return res


if __name__ == '__main__':
    ILRMA = ILRMA()
    mix = np.random.random_integers(-256, 256, [6, 4096*(2**3)])
    mix = mix.astype(np.float32)
    mix = ILRMA.valueprotect(mix/256)
    sep = ILRMA.bss_ILRMA(mix)
    print(sep.shape)
