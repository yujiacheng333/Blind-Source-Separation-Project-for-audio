import numpy as np
import librosa
import os
import re
import shutil
import stat
from sphfile import SPHFile
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()


class get_tfrecord(object):

    def __init__(self, org_train=None, org_test=None):
        if org_train is None:
            self.org_train = "./TIMIT_data/TRAIN/"
        else:
            self.org_train = org_train
        if org_test is None:
            self.org_test = "./TIMIT_data/test/"
        else:
            self.org_test = org_test
        self.record_name = "TIMIT.tfrecords"
        self.path = "./" + self.record_name
        self.batch_size = 32
        self.epo = 5
        self.buffer_size = 500
        self.targ_dir = "./TIMIT_data/audio_only/"
        self.sub_add = "_n"
        self.sr = 16000
        self.action_rate = {
            "random_mask": 0,
            "random_move": 1,
            "random_noise": 0.5
        }
        self.STFT_param = {
            "fft_length": 256 * 4 - 1,
            "frame_step": 256,
            "frame_length": 256 * 4,
            "pad_end": True,
            "name": "stft"
        }
        self.move_file_2_wav_n()
        self.get_TFRecord()
        self.load_tfrecord()


    @staticmethod
    def get_dir_frombase(org):
        dirs = os.listdir(org)
        audios = []
        res_dir = []
        for i in dirs:
            train_sub_dir = os.listdir(org + i)
            for j in train_sub_dir:
                res_dir.append(org + i + '/' + j)
        for i in res_dir:
            last_dir = os.listdir(i)
            for j in last_dir:
                if re.match(".*?.WAV", j):
                    audios.append(i + "/" + j)
        return audios

    def move_file_2_wav_n(self, remove_flag=False):
        if not remove_flag:
            train_videos = self.get_dir_frombase(self.org_train)
            print("fine to load train_data's name, length is {}".format(len(train_videos)))
            test_videos = self.get_dir_frombase(self.org_test)
            print("fine to load test_data's name, length is {}".format(len(test_videos)))
            all_videos = train_videos + test_videos
            if os.path.exists(self.targ_dir):
                if len(os.listdir(self.targ_dir)) or "1" + self.sub_add + ".wav" in os.listdir(targ_dir):
                    print("The file might be exsist this function {} might not work"
                          .format(self.move_file_2_wav_n.__name__))
                    return
            else:
                os.mkdir(self.targ_dir)
            for i, fp in enumerate(all_videos):
                shutil.copy(fp, self.targ_dir + str(i) + ".WAV")
            for i in range(len(os.listdir(self.targ_dir))):
                fp = self.targ_dir + "/" + str(i) + ".wav"
                sph = SPHFile(fp)
                sph.write_wav(filename=fp.replace(".wav", self.sub_add + ".wav"))
                print("fin {}".format(i))
        else:
            a = input("the dir:{} will be remove".format(self.targ_dir))
            if a:
                try:
                    os.chmod(self.targ_dir, stat.S_IWOTH)
                    os.remove(self.targ_dir)
                except PermissionError:
                    print("Permission is dine,after use chomod try to run with sudo")
        return

    @staticmethod
    def zero_padding_seq(x, max_l, action_rate=None, random_move=False):
        if len(x) > max_l:
            return x[0:max_l]
        if not random_move:
            if max_l > len(x):
                pad_l = int((max_l - len(x)) / 2)
                buffer = np.pad(x, (pad_l, pad_l), "constant", constant_values=(x[0], x[-1]))
                if len(buffer) - max_l != 0:
                    buffer = np.append(buffer, x[-1])
                x = buffer
            else:
                if len(x) == max_l:
                    return x
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    x = x[0:max_l]
                else:
                    x = x[len(x) - max_l - 1:-1]
            return x
        else:
            prob = np.random.uniform(0, 1)
            if prob < action_rate:
                if max_l == len(x):
                    return x

                movestep = np.random.randint(low=0, high=max_l - len(x))
                remainstep = max_l - len(x) - movestep
                x = np.append(np.zeros([movestep]), x)
                x = np.append(x, np.ones(remainstep) * x[-1])
                return x
            else:

                pad_l = int((max_l - len(x)) / 2)
                buffer = np.pad(x, (pad_l, pad_l), "constant", constant_values=(x[0], x[-1]))
                if len(buffer) - max_l != 0:
                    buffer = np.append(buffer, x[-1])
                x = buffer
                return x

    @staticmethod
    def random_mask(x, action_rate, min_mask=0, max_mask=0.5):
        prob = np.random.uniform(0, 1)
        if prob < action_rate:
            l = np.random.randint(int(len(x) * min_mask), int(len(x) * max_mask))
            pos = np.random.randint(0, int(len(x) * max_mask - 1))
            x[pos:pos + l] = x[pos]
            return x
        else:
            return x

    @staticmethod
    def get_img(mat):
        mat = np.abs(mat)
        res = np.zeros([mat.shape[0], mat.shape[1], 3])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                res[i, j, :] = np.ones([3]) * mat[i, j]
        plt.imshow(res)
        plt.show()

    @staticmethod
    def random_noise(x, action_rate, max_noise=0.2):
        prob = np.random.uniform(0, 1)
        if prob < action_rate:
            random_leakage = np.random.uniform(0, 1)
            return x + np.random.normal(loc=0, scale=np.var(x) * max_noise * random_leakage, size=x.shape)
        else:
            return x

    @staticmethod
    def min_max_normalize(x):
        z_o_ser = ((x - np.min(x)) / (np.max(x) - np.min(x)))
        return (z_o_ser - np.mean(z_o_ser)) * 2

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def mu_law_encode(auido, quanti):
        mu = float(quanti)
        amp = np.log1p(mu * np.abs(auido)) / np.log1p(mu)
        sig = amp * np.sign(auido)
        return ((sig + 1) / 2 * mu + 0.5).astype(np.int32)

    @staticmethod
    def mu_law_decode(out, quanti):
        mu = float(quanti)
        out = 2 * ((out).astype(np.float) / mu) - 1
        amp = 1 / mu * ((1 + mu) ** np.abs(out) - 1)
        return np.sign(out) * amp


    def get_TFRecord(self, action_rate=None, length_method="max", expand_rate=1,
                     record_name="TIMIT.tfrecords"):
        """
        :param action_rate: 
        method: should in [random_mask, random_move, random_noise] random_noise should 
        be small and with distribution of Normal
        action_rate: the index i should in 0-1 as the prob of action method i
        :param length_method: shold in ["max", "min"]
        :param expand_rate:the expadning of the org data
        :return: output a TFrecord file 
        """""
        sr = self.sr
        sub_add = self.sub_add
        tar_dir = self.targ_dir
        if os.path.exists(self.path):
            print("The record file has all ready exsist")
            return "./" + record_name
        writer = tf.python_io.TFRecordWriter(record_name)
        all_wav_name = glob.glob(tar_dir + "*" + sub_add + ".wav")
        res = []
        for i, dir in enumerate(all_wav_name):
            res.append(len(librosa.load(dir, sr)[0]))
        L = np.asarray(res)

        assert length_method.lower() in ["min", "max"]
        if length_method == "max":
            L = np.max(L)
        else:
            L = np.min(L)
        if (expand_rate * len(res) - expand_rate * len(res)) != 0:
            print("The expanding might not as much as you give")
        data_l = int(expand_rate * len(res))
        counter = 0
        for i in range(data_l):
            if i >= len(all_wav_name) - 1:
                counter = np.random.randint(1, len(all_wav_name) - 1)
            buffer_video, sr = librosa.load(all_wav_name[counter], sr)
            buffer_video = self.min_max_normalize(buffer_video)
            if action_rate is not None:
                if "random_mask" in action_rate:
                    buffer_video = self.random_mask(x=buffer_video, action_rate=action_rate["random_mask"])
                    assert len(buffer_video) > 0
                if "random_move" in action_rate:
                    buffer_video = self.zero_padding_seq(buffer_video, L, action_rate=action_rate["random_move"],
                                                    random_move=True)
                    assert len(buffer_video) > 0
                if "random_noise" in action_rate:
                    buffer_video = self.random_noise(x=buffer_video, action_rate=action_rate["random_noise"])
                    assert len(buffer_video) > 0
                else:
                    buffer_video = self.zero_padding_seq(buffer_video, L)
                    assert len(buffer_video) > 0
            else:
                buffer_video = self.zero_padding_seq(buffer_video, L)
                assert len(buffer_video) > 0
            if i % 1000 == 0:
                plt.plot(buffer_video)
                plt.show()
            buffer_video = self.mu_law_encode(buffer_video, 256)
            if i % 1000 == 0:
                plt.ylim((-1, 1))
                plt.plot(self.mu_law_decode(buffer_video, 256))
                plt.show()
            # seta_mat, amp_mat = np.angle(buffer_video).astype(float), np.abs(buffer_video).astype(np.float)
            buffer_video = buffer_video.tostring()
            # seta_mat = seta_mat.tostring()
            # amp_mat = amp_mat.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "buffer_video": self._bytes_feature(buffer_video),
                    "L": self._int64_feature(L)
                }
            ))

            writer.write(example.SerializeToString())
            counter += 1
        writer.close()

    def make_stft(self, test_d, STFT_params):
        test_d = self.mu_law_decode(test_d.numpy(), 256)
        test_d = tf.cast(test_d, tf.float32)
        test_d = tf.signal.stft(test_d, **STFT_params)
        return test_d

    def load_tfrecord(self):
        path = self.path
        assert isinstance(path, str)
        raw_dataset = tf.data.TFRecordDataset(path)
        description_data = {
            "L": tf.FixedLenFeature([1], tf.int64),
            "buffer_video": tf.FixedLenFeature([], tf.string)
        }
        parsed_video_dataset = raw_dataset.map(lambda x: tf.parse_single_example(x, description_data))
        gd_dataset = parsed_video_dataset.map(lambda x: (x["L"], tf.decode_raw(x['buffer_video'], tf.int32)))
        gd_dataset = gd_dataset.shuffle(self.buffer_size).repeat(self.epo).batch(self.batch_size)
        gd_dataset_iter = gd_dataset.make_one_shot_iterator()
        batch_L, batch_audio = gd_dataset_iter.get_next()
        L = batch_L[0].numpy()
        print(batch_audio.shape)
        print(L)
        self.iter = gd_dataset_iter
        self.L = L

    def get_next_batch(self):
        iter = self.iter
        STFT_params = self.STFT_param
        next_value = iter.get_next()[1]
        res = []
        for i in range(next_value.shape[0]):
            res.append(self.make_stft(next_value[i, :], STFT_params))
        res = tf.cast(res, tf.float32)
        return res

    def main(self):
        """
        for test only
        signals, frame_length, frame_step, fft_length=None,
             window_fn=window_ops.hann_window,
             pad_end=False, name=None
        :return:
        """

        # path = get_TFRecord(sub_add=sub_add, tar_dir=tar_dir,action_rate=action_rate, expand_rate=2)

        print(self.get_next_batch().shape)

if __name__ == '__main__':
    a = get_tfrecord()
    print(a.get_next_batch())