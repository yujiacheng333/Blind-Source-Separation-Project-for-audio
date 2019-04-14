
from scipy.io import wavfile
import os
import librosa
from Controller import json_reader
import pygame


class FileIO:
    """read load write play wav files
    """
    def __init__(self):
        dic_reader = json_reader.JsonReader()
        self.dic = dic_reader.load_json()
        self.dir = self.dic["file_path"]
        self.sr = 0
        self.files = os.listdir(self.dir)
        print("num of files is {}".format(len(self.files)))
        print("Declear Ops file_path is {}".format(self.dir))

    def loadvideo(self, filename):
        """

        :param filename: the file name in self.dir
        :return:  the np array
        """
        loaddir = self.dir + "/" + filename
        sr, video = librosa.load(loaddir)
        self.sr = sr
        return video

    def writevideo(self, video):
        """

        :param video:  the np array to save
        :return: None
        """
        num = len(self.files)
        file_name = self.dir + "/" + str(num) + ".wav"
        wavfile.write(data=video, rate=self.sr, filename=file_name)

    def playmusic(self, filename, loops=0, start=0.0, value=0.5):
        """
        :param filename: 文件名
        :param loops: 循环次数
        :param start: 从多少秒开始播放
        :param value: 设置播放的音量，音量value的范围为0.0到1.0
        :return:
        """
        flag = False
        pygame.mixer.init()
        while 1:
            if flag == 0:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play(loops=loops, start=start)
                pygame.mixer.music.set_volume(value)
            if pygame.mixer.music.get_busy():
                flag = True
            else:
                if flag:
                    pygame.mixer.music.stop()
                    break


if __name__ == '__main__':
    a = FileIO()
    video = a.loadvideo("t0000.wav")
