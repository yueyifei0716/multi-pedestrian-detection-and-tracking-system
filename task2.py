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
RESULT_PATH = 'task2.mp4'

drawing = False
mouse_x1, mouse_y1, mouse_x2, mouse_y2 = -1, -1, -1, -1

def draw_rect(event, x, y, flags, param):
    global mouse_x1, mouse_y1, mouse_x2, mouse_y2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mouse_x1, mouse_y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            mouse_x2, mouse_y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mouse_x2, mouse_y2 = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        mouse_x1, mouse_y1, mouse_x2, mouse_y2 = -1, -1, -1, -1

cv2.namedWindow('Task 2')
cv2.setMouseCallback('Task 2', draw_rect)

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
    old_man = 0
    all_man = 0
    old_man_id = None

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

        number_in_rect = 0
        # process each pedestrian with its bounding box
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            if pos_id not in man_id:
                man_id.append(pos_id)
                points[str(pos_id)] = [[(x1+x2)/2, (y1+y2)/2]]
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

            # get the coordinates of the rectangle drawn in the image
            m_x1 = min(mouse_x1, mouse_x2) * 1080 / 500
            m_x2 = max(mouse_x1, mouse_x2) * 1080 / 500
            m_y1 = min(mouse_y1, mouse_y2) * 1080 / 500
            m_y2 = max(mouse_y1, mouse_y2) * 1080 / 500

            # check if each bounding box for each pedestrian is within that rectangular region
            if m_x1<((x1+x2)/2) and ((x1+x2)/2)<m_x2 and m_y1<((y1+y2)/2) and ((y1+y2)/2)<m_y2:
                number_in_rect += 1

        # the total count of pedestrians in the current frame
        now_man = len(bboxes)

        # calculate each total count for pedestrians
        if old_man == 0:
            all_man = now_man
            old_man = now_man
            old_man_id = man_id.copy()

        # bug here: frame 1: 12345 frame 2:12356
        if now_man > old_man:
            c = len(man_id) - len(old_man_id)

            if c > 0:
                all_man += c
        print('man_id:', man_id)

        font = cv2.FONT_ITALIC
        # report the total count of pedestrians present in the current video frame
        cv2.putText(image, f"now man num:{now_man}", (10, 30), font, 1,
                    [0, 0, 255], 2)
        # report the total count of all unique pedestrians detected since the start of the video.
        cv2.putText(image, f"all man num:{all_man}", (10, 70), font, 1,
                    [0, 0, 255], 2)
        # report the total count of pedestrians who are currently within the manually drawn region
        cv2.putText(image, f"rect man num:{number_in_rect}", (10, 110), font, 1,
                    [0, 0, 255], 2)

        old_man = now_man
        old_man_id = man_id.copy()

        # resize the output frame
        result = imutils.resize(image, height=500)
        # draw the rectangular region within the current video window after resized
        cv2.rectangle(result, (mouse_x1, mouse_y1), (mouse_x2, mouse_y2), (0, 255, 0), 2)

        # save the output video
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))
        videoWriter.write(result)

        # show the output video
        cv2.imshow('Task 2', result)
        cv2.waitKey(t)
        # press 'q' on the keyboard to terminate the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()