"""
copyright from Daichi Kitamura,the org paper is
http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf
Author Daichi Kitamura1
And reference code is from https://github.com/d-kitamura/ILRMA/tree/master/ILRMA_release20180411
for python version is made by yujiacheng333 bupt
concat me: yujiacheng333@bupt.edu.cn
"""
import numpy as np
import json_reader


class ILRMAavoidna(object):
    """
        the implementation of ILRMA1 in org paper
        this is to avoid NAN when test
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

    def ilrmabody(self, X,T = None,V = None ):
        """
        the No.1 step of ILRMA1
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
            buffer = int(t/10)
            if (buffer - t) > 0.5:
                self.n_base = buffer + 1
            else:
                self.n_base = buffer
        if n_channels > t:
            raise ValueError('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).')
        W = np.zeros([self.n_source, n_channels, f], dtype=np.float32)
        W = self.valueprotect(W)
        if W.shape[0] != W.shape[1]:
            W = self.valueprotect(np.random.uniform(low=-0.5, high=0.5, size=W.shape))
            W = W + self.valueprotect(np.random.uniform(low=-0.5, high=0.5, size=W.shape))*1j
        else:
            for i in range(len(W[0, 0, :])):
                W[:, :, i] = np.eye(n_channels)
        if not (T and V):
            T = np.random.uniform(low=0, high=1, size=[f, self.n_base, self.n_source])
            V = np.random.uniform(low=0, high=1, size=[self.n_base, t, self.n_source])
        R = np.zeros([f, t, self.n_source])
        Y = R
        for i in range(f):
            Y[i, :, :] = np.transpose(np.matmul(W[:, :, i], np.transpose(X[i, :, :].squeeze())))
            # shape is [f, n_channel, n_source]
        P = np.power(np.abs(Y), 2)
        E = np.eye(self.n_source)
        Xp = np.transpose(X, [2, 1, 0])
        cost = np.zeros([self.Iter+1, 1])

        for n in range(self.n_source):
            R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
        UN1 = np.ones([self.n_source, 1])  # unit vector for expand the single value shape = [self.n_source,1]
        U1J = np.ones([1, t])  # unit vector for expand the single value shape = [1, t]
        cost = np.zeros([self.Iter + 1, 1])
        for it in range(self.Iter):
            cost[0, 0] = self.cost_function_local(P, R, W, f, t)
            for n in range(self.n_source):
                # update T
                print(n)
                P_R = P[:, :, n] / np.power(R[:, :, n], 2)
                Rhat_VT = np.matmul(np.linalg.inv(R[:, :, n]), np.transpose(V[:, :, n]))
                P_R_VT = np.matmul(P_R , np.transpose(V[:, :, n]))
                T[:, :, n] = T[:, :, n] / np.sqrt(P_R_VT / Rhat_VT)

                T[:, :, n] = self.valueprotect(T[:, :, n])
                R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
                V[:, :, n] = V[:, :, n] / np.sqrt(np.matmul(np.transpose(T[:, :, n]),
                                                            (P[:, :, n] / np.power(R[:, :, n], 2))) /
                                                  np.matmul(T[:, :, n].T, np.linalg.inv(R[:, :, n])))
                R[:, :, n] = np.matmul(T[:, :, n], V[:, :, n])
                for i in range(f):
                    U = (np.matmul(Xp[:, :, i] * np.matmul(UN1, (U1J / R[i, :, n])), np.transpose(Xp[:, :, i])))/t
                    # U is the M, M mat M is the length = n_channel
                    w = np.matmul(np.linalg.inv(np.matmul(W[:, :, i], U)), E[:, n])
                    w /= np.matmul(np.matmul(np.sqrt(np.transpose(w)), U), w)
                    W[n, :, n] = np.transpose(w)
                    Y[i, :, n] = np.matmul(W[n, :, n], Xp[:, :, i])
            P = np.power(np.abs(Y), 2)
            P = self.valueprotect(P)
            if self.normalize:
                zeta = np.sqrt(np.sum(np.sum(P, axis=1), axis=2)) / (t * f)
                zeta_n = zeta.squeeze()
                d = np.expand_dims(self.repmat(zeta_n, [n_channels, f]), axis=0)
                W /= d
                zetaIJ = np.power(np.expand_dims(self.repmat(zeta_n, [f, t]),axis=-1), 2)
                P = P / zetaIJ
                R = R / zetaIJ
                zetaIL = np.power(np.expand_dims(self.repmat(zeta_n, [f, t]), axis=-1), 2)
                T = T / zetaIL
            cost[it+1, 0] = self.cost_function_local(P, R, W, f, t)
        return P, R, W, f, t

    @staticmethod
    def cost_function_local(P, R, W, f, t):
        """
        caculate loss of LIRMA at the org code line 300
        :param P: the powspectgram of Y
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
                x = - np.inf
            A[i] = np.log(x)
        return np.sum(np.sum(np.sum(P/R + np.log(R), axis=2), axis=1), axis=0) - 2 * t * np.sum(A)

    @staticmethod
    def valueprotect(X):
        """

        :param X: arbitrary mat to avoid underflow
        :return: the fixed mat
        """
        mask = X < 1e-12
        X[mask] = 1e-12
        mask = X > 1e12
        X[mask] = 1e12
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


if __name__ == '__main__':
    ILRMA1 = ILRMAavoidna().ilrmabody(np.ones([100,100,3]))