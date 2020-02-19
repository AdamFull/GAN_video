import cv2, os, re
import tensorflow as tf
import numpy as np
import json, psutil, math, pickle

from util import Visualization

vis = Visualization()

class dataset:
    def __init__(self, tpath,  fpath="", width=256, heigth=256, icolor=cv2.COLOR_BGR2GRAY):
        self.tpath = tpath
        self.dpath = os.path.join(self.tpath, "dumps")
        self.folder_names = ["train"]
        self.width = width
        self.height = heigth
        self.icolor = icolor
        #self.create_dir(self.tpath)
        self.process = psutil.Process(os.getpid())

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def memsize(self):
        memory_usage = int(self.process.memory_info().rss)
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(memory_usage, 1024)))
        p = math.pow(1024, i)
        s = round(memory_usage / p, 2)
        return s, size_name[i]
    
    def dump_memory(self, dumping_var, dump_number):
        #self.create_dir(self.dpath)
        with open(os.path.join(self.dpath, "dump%s.txt" % dump_number), "wb") as dump:
            pickle.dump(dumping_var, dump)
            dump.close()

    def prepare(self):
        '''Create image dataset from video'''
        video = cv2.VideoCapture(self.fpath)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Detected %s frames. Starting convertation video to frames." % length)
        data = []
        for i in range(length):
            used_size, size_name = self.memsize()
            success,image = video.read()
            vis.print_progress_bar(i+1, length, label="RAM USAGE %s %s" %  (used_size, size_name), points=70)
            try:
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, self.icolor)
            except cv2.error:
                continue
            image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
            data.append(image)
        print("")
        return np.asarray(data)
    
    def prep_imgs(self, path):
        imgs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        data = []
        for i in range(len(imgs)):
            used_size, size_name = self.memsize()
            vis.print_progress_bar(i+1, len(imgs), label="RAM USAGE %s %s" %  (used_size, size_name), points=70)
            try:
                image = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            except cv2.error:
                continue
            data.append(image)
        print("")
        return np.asarray(data)

    def save_data_record(self):
        '''Save record from image data'''
        print("Restoring and saving record. This may take several minutes.")
        restored = []
        dumps = [os.path.join(self.dpath, f) for f in os.listdir(self.dpath) if os.path.isfile(os.path.join(self.dpath, f))]
        dumps.sort(key=lambda f: int(re.sub('\D', '', f)))
        for i in range(len(dumps)):
            used_size, size_name = self.memsize()
            vis.print_progress_bar(i+1, len(dumps), label="RAM USAGE %s %s" %  (used_size, size_name), points=70)
            with open(dumps[i], "rb") as dump:
                restored += pickle.load(dump)

        return
    
    def load_data_record(self):
        with open(os.path.join(self.tpath, "dataset.tf"), "r") as f:
            data_array = json.load(f)
        return np.asarray(data_array["data"])
