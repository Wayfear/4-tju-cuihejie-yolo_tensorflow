from os import listdir
from os.path import isfile, join

import cv2
import xml.etree.cElementTree as ET
import numpy as np

import config as cfg


class VocProvider(object):

    def __init__(self, args):
        self.data_dir = cfg.DATA_PATH
        self.jpeg_dir = self.data_dir + 'JPEGImages/'
        self.label_dir = self.data_dir + 'Annotations/'
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.box_per_cell = cfg.BOX_PER_CELL

        self.classes = cfg.CLASSES
        self.class_num = len(cfg.CLASSES)
        self.batch_num = cfg.BATCH_NUM

        self.cursor = 0
        self.labels = []

        self._init_data()

    def _init_data(self):

        for file_name in listdir(self.jpeg_dir):
            root = ET.ElementTree(file=self.label_dir + file_name).getroot()
            size_ele = root.find('size')
            width = float(size_ele.find('width').text)
            height = float(size_ele.find('height').text)
            w_scale = self.image_size / width
            h_scale = self.image_size / height

            data = np.zeros([self.cell_size, self.cell_size, 5 + self.class_num], np.float32)

            for obj_ele in root.findall('object'):
                class_name = obj_ele.find('name').text
                class_index = self.classes.index(class_name)
                xmin = (float(obj_ele.find('xmin').text) - 1) * w_scale
                ymin = (float(obj_ele.find('ymin').text) - 1) * h_scale
                xmax = (float(obj_ele.find('xmax').text) - 1) * w_scale
                ymax = (float(obj_ele.find('ymax').text) - 1) * h_scale

                boxes = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
                x_index = int(boxes[0] * self.cell_size / self.image_size)
                y_index = int(boxes[1] * self.cell_size / self.image_size)

                if label[y_index, x_index, 0] == 1:
                    continue

                label[y_index, x_index, 0] = 1
                label[y_index, x_index, 1:5] = boxes
                label[y_index, x_index, 5 + class_index] = 1

            label = {'imname': file_name, 'data': data}
            self.labels.append(label)

    def get_data(self):
        """Get train data

        Returns:
            images: [?, IMAGE_SIZE, IMAGE_SIZE, 3]
            labels: [?, IMAGE_SIZE, IMAGE_SIZE, 5 + CLASS_NUM]
        """

        # for i in range(64):
