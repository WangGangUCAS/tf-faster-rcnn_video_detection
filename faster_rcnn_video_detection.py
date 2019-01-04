import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import  os, sys, cv2
import argparse
import tensorflow as tf
from nets.vgg16 import vgg16

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, '{}>= {:.1f}'.format(class_name, thresh), (int(bbox[0]), int(bbox[3])), font, 1, (0, 255, 0), 2)
    cv2.rectangle(im, (int(bbox[0]), int(bbox[3])), (int(bbox[2]), int(bbox[1])), (0, 255, 0), 5)
    cv2.imshow("im", im)


def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print(('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
    frameRate = 1.0 / timer.total_time
    print("fps: " + str(frameRate))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        cv2.putText(im, '{:s} {:.2f}'.format("FPS:", frameRate), (1750, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    tfmodel = "/home/gpu/tf-faster-rcnn-master/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt"
    args = parse_args()
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    net = vgg16()
    net.create_architecture("TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = im_detect(sess, net, im)

    videoFilePath = '/home/gpu/2018_1030_072827.MP4'
    videoCapture = cv2.VideoCapture(videoFilePath)
    # success, im = videoCapture.read()
    while True:
        success, im = videoCapture.read()
        demo(sess, net, im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    videoCapture.release()
    cv2.destroyAllWindows()

