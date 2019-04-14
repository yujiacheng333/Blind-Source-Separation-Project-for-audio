import numpy as np
import librosa
from Controller import json_reader
import tensorflow as tf


class OPS:
    """the caculation class
    license to:Yujiacheng BUPT
    private file: others not open
    concatMe yujiacheng333@bupt.edu.cn
    """
    def __init__(self):
        self.params = json_reader.JsonReader().load_json()
        self.length = self.params["length"]
        self.max_amp = self.params["max_amp"]

    def trim_silence(self, audio, threshold, frame_length=2048):
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

    def audio2mat(self, list_audio):
        """

        :param list_audio: [audio1, audio2, audio3,....]
        :return: nparray->[[audio1],[audio2],[audio3].....]
        """
        list_audio = np.asarray(list_audio)
        print(list_audio.shape)
        return list_audio

    def normalize(self, audios, type="W",):
        """
        :type: whiting or min-max normalize input should be W or M
        :param audios: np->matrix of
        :return:
        """
        type = type.lower()
        if len(audios.shape) != 2:
            raise ValueError("list_audio: [audio1, audio2, audio3,....]")
        if type in ["w", "m"]:
            if type== "w":
                for i,audio in enumerate(audios):
                    audios[i] = (audio - np.min(audio))/(np.max(audio) - np.min(audio))
            else:
                for i,audio in enumerate(audios):
                    audios[i] = (audio - np.mean(audio))/np.var(audio)
        else:
            raise ValueError("type: whiting or min-max normalize input should be W or M")

    def mu_law_encode(self, audio, quantization_channels):
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

    def mu_law_decode(self, output, quantization_channels):
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

    def perp_mu(self, audio):
        """
        ready for mulaw encoder
        :param audio: 0~1 range audio
        :return: -1~1 range audio
        """
        return (audio-0.5)*2

    def stft(self, audio):
        """
        the params should be written in json files
        :param audio: quantizaed audio
        :return: angle mat, amp mat of TFbins of each audio
        """
        tf_mat = librosa.stft(audio,
                              self.params["fft_sz"],
                              self.params["move_length"],
                              self.params["win_length"],
                              self.params["win_type"],)
        amp_mat = np.abs(tf_mat)
        angle_mat = np.angle(tf_mat)
        return amp_mat, angle_mat

    def istft(self,amp_mat, angle_mat):
        """
        the inverse of stft params should be written in json files
        :param amp_mat: the return of stft
        :param angle_mat: the return of stft
        :return: the rec audio
        """
        tf_mat = amp_mat*np.exp(1j*angle_mat)
        audio = librosa.istft(tf_mat,
                              self.params["move_length"],
                              self.params["win_length"],
                              self.params["win_type"])
        return audio

    def timetobatch(self, audio, batch):
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

    def batchtotime(self, value, batch):
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

    def iva(self, inputaudios):
        """
        the implemete of IVA the heiper params are read from json file
        :param inputaudios:
        :return:
        """
        num_source = self.params["num_source"]
        for audio in inputaudios:
            return None

    def nmf(self, inputaudios):
        """
        the implemete of NMF the heiperparams are read from json file
        :param inputaudios:
        :return:
        """
        num_source = self.params["num_source"]
        for audio in inputaudios:
            return None
