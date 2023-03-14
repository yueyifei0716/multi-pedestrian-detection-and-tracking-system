import numpy as np
import imutils
from objdetector import Detector
import cv2
import random
import os
from pathlib import Path
import motmetrics as mm

# allow repeated loading of dynamic link libraries
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# paths for input and output videos
VIDEO_PATH = './data/video/video_0009.mp4'


if __name__ == '__main__':

    # # open the input video
    # cap = cv2.VideoCapture(VIDEO_PATH)
    # # initialize deepsort detector
    # det = Detector()
    # # get the frame per second of the input video
    # fps = int(cap.get(5))
    # print('The fps is :', fps)
    # t = int(1000 / fps)
    # # get the total frames of the input video
    # frames_num = int(cap.get(7))
    # print('The total frame is: ', frames_num)

    # # initialize some variable
    # videoWriter = None
    # man_id = list()
    # points = dict()
    # colors = dict()
    # frame_id = 1
    # preds = []
    # while True:
    #     # grab a single frame of video
    #     _, im = cap.read()
    #     # check if the current frame is existing
    #     if im is None:
    #         break
    #     # detect all pedestrians in all frames and calculate the bounding box for each of them
    #     image, bboxes = det.feedCap(im)
    #     if bboxes != []:
    #         for bbox in bboxes:
    #             x1, y1, x2, y2, cls_id, bbox_id = bbox
    #             bb_left = x1
    #             bb_top = y1
    #             bb_width = x2 - x1
    #             bb_height = y2 - y1
    #             preds.append([frame_id, bbox_id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1])
    #     print('Predited ' + str(frame_id))
    #     frame_id += 1


    # output_path = Path('./data/prepare_train_data/')

    # with open(output_path / 'predit.txt', 'w', encoding='utf-8') as f:
    #     for i in preds:
    #         f.write(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + ',' + str(i[3]) + ',' + str(i[4])
    #                    + ',' + str(i[5]) + ',' + str(i[6]) + ',' + str(i[7]) + ',' + str(i[8]) + ',' + str(i[9]) + '\n')
    #     f.close()

    gt_file = './data/prepare_train_data/0009_bbox_labels/ground_truth.txt'
    ts_file = './data/prepare_train_data/predit.txt'
    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D")  # 读入GT
    ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.9)
    mh = mm.metrics.create()
    print(mm.metrics.motchallenge_metrics)
    summary = mh.compute(acc, metrics=['precision', 'recall', 'mota'], name='Overall')

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={'precision': 'Percision', 'recall' : 'Recall', 'mota': 'MOTA'}
    )
    print()
    print('The evaluation results:')
    print(strsummary)