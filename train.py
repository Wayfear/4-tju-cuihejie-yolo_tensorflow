import tensorflow as tf

from yolo_net.yolo_v1 import Yolo

def main():
    yolo = Yolo()
    inputs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    net = yolo._build_net(inputs, 10, 2, 7)
    print("Done")

if __name__ == '__main__':
    main()