import numpy as np
import json_reader
import os
import librosa
import ILRMA
import OPS
from scipy.io import wavfile
import matplotlib.pyplot as plt
# test only


class InterFaceMain(object):
    """
    the main class of controller
    to accept the call back from front
    """
    def __init__(self):
        """it will be build after controller finished"""
        js = json_reader.JsonReader()
        self.params  = js.load_json()
        self.fsresample = self.params["fsresample"]
        if self.fsresample < 8000:
            raise Warning("Low sample rate will inflect the performance of the project")

    def testcontroller(self):
        """
        the test func for controller
        :param mix: the mixture of mutichannel signals
        :return: the separated signal
        """
        sig = []
        test_file_path = "../input"
        files = os.listdir(test_file_path)
        for i, f in enumerate(files):
            f = test_file_path + "/" + f
            sr, buffer = wavfile.read(f)
            buffer_sig = np.asmatrix(buffer).T.astype(np.float32)
            buffer_sig = librosa.resample(buffer_sig, sr, self.fsresample)
            sig.append(buffer_sig.T)
        sig = np.transpose(np.array(sig), [1, 2, 0])  # shape is signal x channel x source
        mix = np.zeros([sig.shape[0], 2])
        mix[:, 0] = sig[:, 0, 0] + sig[:, 0, 1]
        mix[:, 1] = sig[:, 1, 0] + sig[:, 1, 1]
        mix = OPS.OPS().normalize(mix.T).T
        if np.abs(np.max(mix)) > 1:
            print("Clipped detected while mixing")
        [sep, cost] = ILRMA.ILRMA().bss_ILRMA(mix_audio=mix)
        print(sep.shape)
        plt.subplot(2,1,1)
        plt.plot(sep[0, :])
        plt.subplot(2,1,2)
        plt.plot(sep[1, :])
        plt.show()
        if not os.path.exists("out_dir"):
            os.mkdir("out_dir")
        wavfile.write("./out_dir/out_sep1.wav", self.fsresample, sep[0, :])
        wavfile.write("./out_dir/out_sep2.wav", self.fsresample, sep[1, :])


if __name__ == '__main__':
    InterFaceMain().testcontroller()