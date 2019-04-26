import numpy as np
import scipy.io as sio


class MVDR(object):
    def __init__(self):
        self.n_mac = 6
        self.lamda = 17
        self.r = 0.1
        self.max_pos = np.zeros([3, ])
        self.max_pos[2] = -np.inf

    def MVDR_body(self, X, seta=None, fi = None):
        n = self.n_mac
        Rxx = np.matmul(X, X.T.conj())/self.n_mac
        Rxx_hat = np.linalg.pinv(Rxx)
        if not (seta and fi):
            range_seta = np.arange(n*60-120, (n+2)*60-120, 2)
            range_fi = np.arange(0, 90, 2)
            for i in range_seta:
                for j in range_fi:
                    Aa = self.getAa(i, j)
                    ps = 1/np.matmul(np.matmul(Aa.conj().T, Rxx_hat), Aa)
                    if self.max_pos[2] < ps[0, 0]:
                        self.max_pos[0] = i
                        self.max_pos[1] = j
                        self.max_pos[2] = np.real(ps[0, 0])
        seta = self.max_pos[0]
        fi = self.max_pos[1]
        p = self.max_pos[2]
        Aa = self.getAa(seta, fi)
        Wopt = np.matmul(Aa.conj().T, Rxx_hat) / p
        return np.matmul(Wopt, X).squeeze()


    def getAa(self, i, j):
        res = []
        for n in range(self.n_mac):
            seta_buffer = 2*np.pi*(n-1)/self.n_mac
            k = 2*np.pi/self.lamda
            buffer = -k*self.r*(np.cos(seta_buffer)*np.cos(i)*np.sin(j)+np.sin(seta_buffer)*np.sin(i))
            buffer = np.exp(buffer*1j)
            res.append(buffer)
        return np.expand_dims(np.asarray(res), -1)

    @staticmethod
    def crr(X1, X2):
        crr = []
        t = len(X1)
        for i in range(100):
            X1 = X1[0:t-i]
            X2 = X2[i:t]
            print(X1.shape)
            print(X2.shape)
            crr.append(np.matmul(np.expand_dims(X1, 0), np.expand_dims(X2, -1)).squeeze())
        return np.asarray(crr)


if __name__ == '__main__':
    X = sio.loadmat("./1.mat")["buffer"][0:1024]
    crr = MVDR.crr(X[:, 0], X[:, 1])
