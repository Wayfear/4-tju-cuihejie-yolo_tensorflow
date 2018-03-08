
DATA_PATH = 'VOCdevkit/VOC2007/'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

#
# model parameter
#

BATCH_SIZE = 64

IMAGE_SIZE = 448

CELL_SIZE = 7

BOX_PER_CELL = 2

COORD_SCALE = 5.0
NOOBJ_SCALE = 0.5


