from os import listdir
from os.path import isfile, join

import cv2
import xml.etree.cElementTree as ET
import numpy as np

import config as cfg


class DataProvider(object):

    def __init__(self):
        self.jpeg_dir = cfg.DATA_PATH + 'JPEGImages/'
        self.label_dir = cfg.DATA_PATH + 'Annotations/'

        self.cursor = 0
        self.gl_labels = []

        self._init_data()

    def _init_data(self):

        for file_name in listdir(self.jpeg_dir):
            root = ET.ElementTree(file=self.label_dir + file_name).getroot()
            size_ele = root.find('size')
            width = float(size_ele.find('width').text)
            height = float(size_ele.find('height').text)
            w_scale = cfg.IMAGE_SIZE / width
            h_scale = cfg.IMAGE_SIZE / height

            data = np.zeros([cfg.CELL_SIZE, cfg.CELL_SIZE, 5 + cfg.CLASS_NUM], np.float32)

            for obj_ele in root.findall('object'):
                class_name = obj_ele.find('name').text
                class_index = cfg.CLASSES.index(class_name)
                xmin = (float(obj_ele.find('xmin').text) - 1) * w_scale
                ymin = (float(obj_ele.find('ymin').text) - 1) * h_scale
                xmax = (float(obj_ele.find('xmax').text) - 1) * w_scale
                ymax = (float(obj_ele.find('ymax').text) - 1) * h_scale

                boxes = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
                x_index = int(boxes[0] * cfg.CELL_SIZE / cfg.IMAGE_SIZE)
                y_index = int(boxes[1] * cfg.CELL_SIZE / cfg.IMAGE_SIZE)

                if label[y_index, x_index, 0] == 1:
                    continue

                label[y_index, x_index, 0] = 1
                label[y_index, x_index, 1:5] = boxes
                label[y_index, x_index, 5 + class_index] = 1

            label = {'imname': file_name, 'data': data}
            self.gl_labels.append(label)

        self.train_size = len(self.gl_labels)

    def get_data(self):
        """Get train data

        Returns:
            images: [?, IMAGE_SIZE, IMAGE_SIZE, 3]
            labels: [?, IMAGE_SIZE, IMAGE_SIZE, 5 + CLASS_NUM]
        """

        images = np.zeros((cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), np.float32)
        labels = np.zeros((cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 5 + cfg.CLASS_NUM), np.float32)

        for i in range(cfg.BATCH_SIZE):
            label = labels[self.cursor]
            self.cursor += 1

            impath = self.jpeg_dir + label['imname']
            image = cv2.imread(impath)
            image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

            images[i] = image
            labels[i] = label

        return images, labels
