import numpy as np
import librosa
from scipy import signal
from Controller import json_reader
import tensorflow as tf


class OPS(object):
    """the caculation class
    license to:Yujiacheng333 BUPT
    private file: others not open
    concatMe yujiacheng333@bupt.edu.cn
    """
    def __init__(self):
        self.params = json_reader.JsonReader().load_json()
        self.max_amp = self.params["max_amp"]
        self.angle_mat = None

    @staticmethod
    def trim_silence(audio, threshold, frame_length=2048):
        """
        cut off slience bound in audio
        :param audio: the numpy array read from librosa
        :param threshold: the rate of lowestpow in both bind
        :param frame_length: the length of each frame
        :return: the np array with out low pow bound
        """
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rmse(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    @staticmethod
    def audio2mat(list_audio):
        """

        :param list_audio: [audio1, audio2, audio3,....]
        :return: nparray->[[audio1],[audio2],[audio3].....]
        """
        list_audio = np.asarray(list_audio)
        print(list_audio.shape)
        return list_audio

    @staticmethod
    def normalize(audios, method="m"):
        """
        input should be numpy array [n_sourcr , audio]
        :param audios: whiting or min-max normalize input should be W or M
        :param method: np->matrix of "W"
        :return:
        """
        method = method.lower()
        if len(audios.shape) != 2:
            raise ValueError("list_audio: [audio1, audio2, audio3,....]")
        if method in ["w", "m"]:
            res = np.zeros(audios.shape)
            if method == "m":
                for i, audio in enumerate(audios):
                    res[i, :] = audio * 2/(np.max(audio) - np.min(audio))
            else:
                for i, audio in enumerate(audios):
                    res[i, :] = (audio - np.mean(audio))/np.var(audio)
            return res
        else:
            raise ValueError("type: whiting or min-max normalize input should be W or M")

    @staticmethod
    def mu_law_encode(audio, quantization_channels):
        """
        Quantizes waveform amplitudes.
        :param audio:-1~1 range audio
        :param quantization_channels the number of quantization value
        """
        with tf.name_scope('encode'):
            mu = tf.to_float(quantization_channels - 1)
            # Perform mu-law companding transformation (ITU-T, 1988).
            # Minimum operation is here to deal with rare large amplitudes caused
            # by resampling.
            safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
            magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
            signal = tf.sign(audio) * magnitude
            # Quantize signal to the specified number of levels.
            res = tf.to_int32((signal + 1) / 2 * mu + 0.5)
            return res

    @staticmethod
    def mu_law_decode(output, quantization_channels):
        """
        Recovers waveform from quantized values.
        :param output:the output of encode
        :param quantization_channels:should be same as encode
        """
        with tf.name_scope('decode'):
            mu = quantization_channels - 1
            # Map values back to [-1, 1].
            signal = 2 * (tf.to_float(output) / mu) - 1
            # Perform inverse of mu-law transformation.
            magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
            return tf.sign(signal) * magnitude

    def denormalize(self, audio):
        """
        -max-amp~max~amp
        recover all audio into
        :param audio: 0-1 range file
        :return: -max-amp~max~amp
        """
        if np.mean(audio)>=1:
            raise ValueError("input should be 0-1 zone")
        else:
            res = audio - 0.5
            return res*self.max_amp*2

    @staticmethod
    def perp_mu(audio):
        """
        ready for mulaw encoder
        :param audio: 0~1 range audio
        :return: -1~1 range audio
        """
        if len(audio.shape) == 1:
            return (audio-0.5)*2
        elif len(audio.shape) == 2:
            res = []
            for buffer in audio:
                res.append((buffer - 0.5) * 2)
            return np.asarray(res)

    def stft(self, audio):
        """
        the params should be written in json files
        :param audio: quantizaed audio
        :return: angle mat, amp mat of TFbins of each audio
        """
        # tf_mat = librosa.core.stft(audio, self.params["fft_sz"], self.params["move_length"],
        #                      self.params["win_length"],
        #                      self.params["win_type"],)
        tf_mat = signal.stft(audio, fs=self.params["fsresample"],
                             window=self.params["win_type"], nperseg=self.params["win_length"],
                             noverlap=self.params["win_length"]-self.params["move_length"])
        return tf_mat

    def istft(self, tf_mat):
        """
        the inverse of stft params should be written in json files
        :param tf_mat: the return of stft
        :return: the rec audio
        """
        audio = signal.istft(tf_mat, fs=self.params["fsresample"],window=self.params["win_type"]
                             , nperseg=self.params["win_length"],noverlap=self.params["win_length"]-self.params["move_length"])
        return audio[1].real

    @staticmethod
    def timetobatch(audio, batch):
        """
        used to slice the time seq into littile batch for caculation
        :param audio: the org audio signal can process mutipul when the last dim is audios num
        :param batch: the num to be slice
        :return: [batch,int(audio.shape[0]/batch),audioshape[1]]
        """
        shape = audio.shape
        if len(shape) == 1:
            return np.reshape(audio, [batch, -1])
        if len(shape) == 2:
            return np.reshape(audio, [batch, -1, audio.shape[1]])
        else:
            raise ValueError("dim should be 2")

    @staticmethod
    def batchtotime(value, batch):
        """

        :param value: output of time to batch
        :param batch: batch size
        :return: the seq of audio
        """
        shape = np.shape(value)
        if len(shape) ==2:
            return np.reshape(value, [-1])
        elif len(shape) ==3:
            return np.reshape(value, [-1, shape[1]])
