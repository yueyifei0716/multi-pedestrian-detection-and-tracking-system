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
RESULT_PATH = 'task3.mp4'

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

cv2.namedWindow('Task 3')
cv2.setMouseCallback('Task 3', draw_rect)

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
    old_manid = []
    entered_id = {}
    left_id = {}

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
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness行/字体厚度
        now_ids = []

        number_in_rect = 0
        # process each pedestrian with its bounding box
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            now_ids.append(pos_id)
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

        # all pedestrians are assigned to the corresponding group
        if len(bboxes) > 1:
            groups = [[bboxes[0]]]
            for i in range(1, len(bboxes)):
                person = bboxes[i]
                person_x = (person[0] + person[2]) / 2
                person_y = (person[1] + person[3]) / 2
                person_w = person[2] - person[0]
                person_h = person[3] - person[1]
                person_area = person_w * person_h
                is_found_group = False
                for group in groups:
                    for member in group:
                        print(member)
                        member_x = (member[0] + member[2]) / 2
                        member_y = (member[1] + member[3]) / 2
                        member_w = member[2] - member[0]
                        member_h = member[3] - member[1]
                        member_area = member_w * member_h
                        if 0.7 * member_area < person_area and person_area < 1.3 * member_area:
                            if (member_x - 2 * member_w < person_x and person_x < member_x + 2 * member_w):
                                is_found_group = True
                                group.append(person)
                                break
                    if is_found_group:
                        break
                if not is_found_group:
                    groups.append([person])

            # according to the groups information frame, calculate the number of individuals and groups
            alone_num = 0
            group_num = 0
            for group in groups:
                if len(group) > 1:
                    min_x = group[0][0]
                    min_y = group[0][1]
                    max_x = group[0][2]
                    max_y = group[0][3]
                    member_ids = []
                    for member in group:
                        if member[0] < min_x:
                            min_x = member[0]
                        if member[1] < min_y:
                            min_y = member[1]
                        if member[2] > max_x:
                            max_x = member[2]
                        if member[3] > max_y:
                            max_y = member[3]
                        member_ids.append(member[5])
                    cv2.rectangle(image, (min_x-10, min_y-30), (max_x+10, max_y+10), (255, 255, 0), 2)
                    cv2.putText(image, 'Group{}'.format(str(member_ids)), (min_x-10, min_y-42), 0, tl / 3,
                                [225, 255, 0], thickness=tf, lineType=cv2.LINE_AA)
                    group_num += 1
                else:
                    alone_num += 1
            font = cv2.FONT_ITALIC
            cv2.putText(image, f"alone num:{alone_num}", (1700, 30), font, 1,
                        [0, 0, 255], 2)
            cv2.putText(image, f"group num:{group_num}", (1700, 70), font, 1,
                        [0, 0, 255], 2)

        # the total count of pedestrians in the current frame
        now_man = len(bboxes)
        all_len = len(man_id)

        if not list(set(now_ids) - set(old_manid)) == []:
            entered_id = list(set(now_ids) - set(old_manid))
        if not list(set(old_manid) - set(now_ids)) == []:
            left_id = list(set(old_manid) - set(now_ids))

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
        cv2.putText(image, f"entered id(s): {entered_id}", (10, 150), font, 1,
                    [0, 0, 255], 2)
        cv2.putText(image, f"left id(s): {left_id}", (10, 190), font, 1,
                    [0, 0, 255], 2)

        old_man = now_man
        old_manid = man_id.copy()

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
        cv2.imshow('Task 3', result)
        cv2.waitKey(t)
        # press 'q' on the keyboard to terminate the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
