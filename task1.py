import numpy as np
import imutils
from objdetector import Detector
import cv2
import random
import os

# allow repeated loading of dynamic link libraries
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# paths for input and output videos
VIDEO_PATH = './video/test_07.mp4'
RESULT_PATH = 'task1.mp4'


if __name__ == '__main__':

    # open the input video
    cap = cv2.VideoCapture(VIDEO_PATH)
    # initialize deepsort detector
    det = Detector()
    # get the frame per second of the input video
    fps = int(cap.get(5))
    print('The fps is :', fps)
    t = int(1000 / fps)
    # get the total frames of the input video
    frames_num = int(cap.get(7))
    print('The total frame is: ', frames_num)

    # initialize some variable
    videoWriter = None
    man_id = list()
    points = dict()
    colors = dict()

    while True:
        # grab a single frame of video
        _, im = cap.read()
        # check if the current frame is existing
        if im is None:
            break
        # detect all pedestrians in all frames and calculate the bounding box for each of them
        image, bboxes = det.feedCap(im)
        print(bboxes)
        # define the line/front thickness
        tl = round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

        # process each pedestrian with its bounding box
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            if pos_id not in man_id:
                man_id.append(pos_id)
                points[str(pos_id)] = [[(x1 + x2) / 2, (y1 + y2) / 2]]
                colors[str(pos_id)] = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
            points[str(pos_id)].append([(x1 + x2) / 2, (y1 + y2) / 2])
            c1, c2 = (x1, y1), (x2, y2)

            # draw the bounding box for each pedestrian
            cv2.rectangle(image, c1, c2, (0, 0, 255), thickness=tl, lineType=cv2.LINE_AA)
            # font thickness
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{}-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

            # draw the corresponding trajectory for each pedestrian
            ndarray_pts = np.array(points[str(pos_id)], np.int32)
            image = cv2.polylines(image, [ndarray_pts], isClosed=False, color=colors[str(pos_id)],
                                  thickness=10)

        # resize the output frame
        result = imutils.resize(image, height=500)

        # save the output video
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))
        videoWriter.write(result)

        # show the output video
        cv2.imshow('Task 1', result)
        cv2.waitKey(t)
        # press 'q' on the keyboard to terminate the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
