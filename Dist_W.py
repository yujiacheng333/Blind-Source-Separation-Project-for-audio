import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
fs = 16000
r = 4.7*1e-2
mac_pos = [[1, 0], [1/2, 1.732/2], [-1/2, 1.732/2], [-1, 0], [-1/2, -1.732/2], [1/2, -1.732/2]]
mac_pos = np.asmatrix(mac_pos)*r
aerfa = np.arange(0, 360, 30)
aerfa = np.pi*(aerfa/180)
D = np.zeros([12, 6])
for i, aerfa_local in enumerate(aerfa):
    normal_vec = [np.cos(aerfa_local), np.sin(aerfa_local)]
    for j in range(6):
        projection = np.dot(mac_pos[j, :], normal_vec)
        time_delay = projection / 342
        move = int(time_delay * fs)
        D[i, j] = move
D = D.astype(int)


def move_action(x, stride):
    l = x.shape[0]
    if stride < 0:  # 右移
        x = np.concatenate([np.zeros([abs(stride), ]), np.asarray(x)], axis=0)
        x = x[0:l]
    elif stride > 0:  # 左移
        x = np.concatenate([np.asarray(x), np.zeros([stride, ])], axis=0)
        x = x[-(l+1):-1]
    return x


def get_input_data():
    fs = 16000
    r = 4.7 * 1e-2
    distant = 0
    mac_pos = [[1, 0], [1/2, 1.732/2], [-1/2, 1.732/2], [-1, 0], [-1/2, -1.732/2], [1/2, -1.732/2]]
    mac_pos = np.asarray(mac_pos)
    mac_pos *= r
    aerfa = np.arange(0, 360, 15)
    aerfa = np.pi*(aerfa/180)
    audio = librosa.load("./piano.wav", sr=fs)
    audio = audio[0][1000:47000]
    for aerfa_local in aerfa:
        normal_vec = [np.cos(aerfa_local), np.sin(aerfa_local)]
        audio_local = []
        for j in range(6):
            projection = np.dot(mac_pos[j, :], normal_vec)
            real_dist = projection + distant
            time_delay = real_dist/342
            move = time_delay*fs
            move = move.astype(int)
            out_audio = move_action(audio, -move)
            audio_local.append(out_audio)
        power = []
        for k in range(D.shape[0]):
            buffer = np.zeros(np.asmatrix(audio_local).shape)
            for v in range(6):
                buffer[v] = move_action(audio_local[v], D[k, v])
            yk = np.sum(buffer, axis=0)/6
            wavfile.write("./out"+str(k)+".wav", data=yk, rate=fs)
            power_local = np.linalg.norm(yk[100: -100], 2)
            # power_local = np.max(np.abs(yk))
            power.append(power_local)
        power = np.asarray(power)
        plt.plot(power)
        plt.show()

get_input_data()


