import numpy as np
import tensorflow as tf
import cv2

import config as cfg
from yolo_net.yolo_v1 import Yolo

def main():
    yolo = Yolo()

    ori_img = cv2.imread('test.jpg')
    height, width = ori_img.shape[:2]
    w_scale = cfg.IMAGE_SIZE / width
    h_scale = cfg.IMAGE_SIZE / height

    img = cv2.resize(ori_img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    img = np.expand_dims(img, axis=0)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint("checkpoint"))

    preds = sess.run(yolo.net, feed_dict={yolo.images: img})[0] # [CELL_SIZE, CELL_SIZE, 5 * BOX_PER_CELL + CLASS_NUM]


    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            if preds[i, j, 0] > cfg.THRESHOLD:
                x = preds[i, j, 1] * cfg.IMAGE_SIZE / w_scale
                y = preds[i, j, 2] * cfg.IMAGE_SIZE / h_scale
                w = np.square(preds[i, j, 3]) * cfg.IMAGE_SIZE / w_scale
                h = np.square(preds[i, j, 4]) * cfg.IMAGE_SIZE / h_scale
                box = [
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2))
                ]
                ori_img = cv2.rectangle(ori_img, box[0], box[1], (0, 255, 0), 2)

                class_index = np.argmax(preds[i, j, -cfg.CLASS_NUM:])
                class_name = cfg.CLASSES[class_index]
                cv2.putText(ori_img, class_name, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if preds[i, j, 5] > cfg.THRESHOLD:
                x = preds[i, j, 1] * cfg.IMAGE_SIZE / w_scale
                y = preds[i, j, 2] * cfg.IMAGE_SIZE / h_scale
                w = np.square(preds[i, j, 3]) * cfg.IMAGE_SIZE / w_scale
                h = np.square(preds[i, j, 4]) * cfg.IMAGE_SIZE / h_scale
                box = [
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2))
                ]
                ori_img = cv2.rectangle(ori_img, box[0], box[1], (0, 255, 0), 2)

                class_index = np.argmax(preds[i, j, -cfg.CLASS_NUM:])
                class_name = cfg.CLASSES[class_index]
                cv2.putText(ori_img, class_name, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('image', ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()