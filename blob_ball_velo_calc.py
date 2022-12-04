import glob
import math
import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from codetiming import Timer

from BallSpeedCode.calibration import undistortion_V2, calibrate_frame_1080p

data_path = 'data/'
timestamps = 'timestamps/'


params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.blobColor = 255
# 设置圆度
params.filterByCircularity = True
params.minCircularity = 0.1

simple_blob_detector = cv2.SimpleBlobDetector_create(params)
mog = cv2.createBackgroundSubtractorMOG2(history=8, varThreshold=100, detectShadows=True)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)


def show_resize(path, x, y):
    resieze = cv2.resize(path, (x, y), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Frame", resieze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_blob(blobs, x):
    cv2.rectangle(blobs, (960, 0), (1920, 700), (0, 0, 255), 3, cv2.LINE_AA)  # draw roi
    cv2.circle(blobs, (int(x[0]), int(x[1])), radius=1, color=(0, 0, 255),
               thickness=-1)  # draw ball
    resize = cv2.resize(blobs, (720, 540), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Blobs Using Area ", resize)  # show ROI
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def round_detect(img):
    return simple_blob_detector.detect(img)


def frame_process(frame):  # cant use
    history = 8
    varThreshold = 100
    bShadowDetection = True
    mog = cv2.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)  # 背景前景分離
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 會返回指定形狀和尺寸的結構元素
    fgmask = mog.apply(frame)
    blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
    th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)
    return opening


def blob(video_name, outputDir):
    timestamp = []
    # read_name = timestamps + video_info[0] + '_' + video_info[1] + '_' + video_info[2] + '.csv'
    # with open(read_name, newline='') as csvfile:
    #     rows = csv.reader(csvfile)
    #
    #     for row in rows:
    #         timestamp.append(row[2])
    #
    # write_name = data_path + video_info[0] + '_' +  video_info[1] + '_' + video_info[2] + '_data.csv'
    # file = open(write_name, mode='w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['frame_count', 'cx', 'cy', 'dist', 'frame_rate', 'velo'])

    # last_timestamp = int(timestamp[1])

    tmpVelo = []
    frame_record = []
    lost_frame_record = []
    history = 8
    varThreshold = 100
    bShadowDetection = True
    pixelToMeter = 394.44
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fname = video_name.split('\\')[-1]
    fname = fname.split('.')[0]

    print(fname)
    out = cv2.VideoWriter(outputDir + '/' + fname + '.mp4', fourcc, 30.0, (1920, 1080))  # 720,540   write

    mog = cv2.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)  # 背景前景分離
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 會返回指定形狀和尺寸的結構元素
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frame: ', length)

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    minutes = int(duration / 60)
    seconds = duration % 60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    frame_count = 0
    ret = True
    lastCenter = (0, 0)
    veloCount = 0
    pitchVelo = 0
    frame_record_counter = 0
    velo = 0

    while (ret):
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_count += 1
        # print(ret)

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if frame_count > 250: #timestemps only got 250
        #     break
        if ret and (frame_count > 100 and frame_count < 450):

            try:
                fgmask = mog.apply(frame)
                # if frame_count > 220:
                #     cv2.imshow('fmask',fgmask)
                blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
                th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]

                # if frame_count > 20:
                #     cv2.imshow('th', th)
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)
                # cv2.rectangle(opening, (960, 0), (1920, 700), (255, 255, 255), 3, cv2.LINE_AA)
                # if frame_count > 20:
                #     cv2.imshow('op', opening)

                if frame_count > 100 and frame_count < 450:

                    # tm1=cv2.resize(opening,(720,540),interpolation=cv2.INTER_CUBIC)
                    params = cv2.SimpleBlobDetector_Params()  #

                    params.filterByArea = True
                    params.minArea = 100
                    params.blobColor = 255
                    # 设置圆度
                    params.filterByCircularity = True
                    params.minCircularity = 0.1

                    detector = cv2.SimpleBlobDetector_create(params)
                    keypoints = detector.detect(opening)

                    # print('points: ', keypoints)
                    # print(type(keypoints))

                    blank = np.zeros((1, 1))
                    blobs = cv2.drawKeypoints(opening, keypoints, blank, (0, 0, 255),  ###check result
                                              cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                    try:
                        # print('points: ', keypoints)
                        x_mark = []
                        y_mark = []
                        try:  # multi keypoints
                            # print('frame' , frame_count)

                            for keyPoint in keypoints:
                                x = keyPoint.pt
                                # print("x y: ", x[0],x[1])
                                (cx, cy) = x[0], x[1]  # x,y坐標
                                if (int(x[0]) > 960 and int(x[0]) < 1920) and (int(x[1]) > 0 and int(x[1]) < 700):
                                    # print("x y: ", cx, cy)
                                    frame_record.append(frame_count)  # 記錄可使用的frame
                                    cv2.rectangle(frame, (960, 0), (1920, 700), (0, 0, 255), 3, cv2.LINE_AA)  # draw roi
                                    cv2.circle(blobs, (int(x[0]), int(x[1])), radius=1, color=(0, 0, 255),
                                               thickness=-1)  # draw ball
                                    # cv2.imwrite('sample.jpg',frame)
                                    # resieze = cv2.resize(frame,(720,540),interpolation=cv2.INTER_CUBIC)
                                    # cv2.imshow("Blobs Using Area", frame)   #show ROI
                                    print(frame_count)

                                    veloCount += 1

                                    if (frame_record_counter == 0):
                                        lastCenter = (cx, cy)
                                        # print('INSIDE', frame_count ,frame_record_counter)
                                        frame_record_counter = frame_record_counter + 1

                                    else:
                                        # print(frame_record)
                                        # print(frame_record[frame_record_counter])

                                        if frame_record[frame_record_counter] - frame_record[
                                            frame_record_counter - 1] == 1:
                                            frame_record_counter = frame_record_counter + 1
                                            # print("INSIDE_FRAM_NUMBER: ", frame_count)
                                            diffx = abs(cx - lastCenter[0])  # 兩顆球之間的距離而已
                                            diffy = abs(cy - lastCenter[1])  # 兩顆球之間的距離而已
                                            dist = math.sqrt(diffx ** 2 + diffy ** 2)  # 三角形邊長公式
                                            #         frame_time = int(timestamp[frame_count]) - int(last_timestamp) #兩顆球timestamp的差別
                                            #         if frame_time != 0:
                                            #             frame_rate = 1/(int(timestamp[frame_count]) - int(last_timestamp))*1e9 #變成每秒幾偵
                                            #         else:
                                            #             frame_rate = 0
                                            #
                                            velo = 3600 * (220) * dist / (1000 * pixelToMeter)
                                            # print("inside velo:",velo)
                                            if velo > 60 and velo < 160:
                                                tmpVelo.append(velo)

                                        else:
                                            frame_record_counter = frame_record_counter + 1

                                    # print('frame_rate:',frame_rate)
                                    #
                                    # last_timestamp = int(timestamp[frame_count])
                                    #

                                    #
                                    # print("veloCount, velo: ",veloCount, velo)

                                    # if velo > 60 and velo < 160:  #original place
                                    #     tmpVelo.append(velo)

                                    lastCenter = (cx, cy)
                                    if len(tmpVelo) >= 2:
                                        totalVelo = 0
                                        for oneVelo in tmpVelo:
                                            totalVelo += oneVelo
                                        pitchVelo = totalVelo / len(tmpVelo)
                                        # print("veloCount 3, velo",  pitchVelo, velo)

                        except:  # one keyppints

                            # tu = keypoints[0].pt #pt -> keypoint function
                            # print('keypoints1: ',tu[0],tu[1])
                            pass

                    except:

                        lost_frame_record
                        pass

                # cont, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.RETR_EXTERNAL只取外層 cv2.CHAIN_APPROX_SIMPLE壓縮取回的Contour像素點，只取長寬及對角線的end points
                # if frame_count > 42:
                #     print('cont:',len(cont))  #看抓到了幾個個體

                # for c in cont:
                #     approx = cv2.approxPolyDP(c, .05 * cv2.arcLength(c, True), True) #多邊形逼近，cv2.arcLength：取得Contours周長的指令
                # 最重要的引數就是 epsilon 簡單記憶為：該值越小，得到的多邊形角點越多，輪廓越接近實際輪廓，該引數是一個準確度引數。該函數返回值為輪廓近似多邊形的角點。
                # approx返回值會是多邊形的角點

                # if len(approx) > 3:
                #     (cx, cy), radius = cv2.minEnclosingCircle(c) #minEnclosingCircle會傳回圓心X,Y以及半徑
                #     area = cv2.contourArea(c)
                #     if frame_count > 42:
                #         print('area',area)
                #         cv2.drawContours(frame, c, -1,(0,255,0),3)
                #         cv2.imshow('frame', frame)

                # if (area > 50) and (area < 300):
                #     (x, y, w, h) = cv2.boundingRect(c) # 框出有可能是球的地方
                #     if frame_count > 410:
                #         print(x, y, w, h)
                #         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3,cv2.LINE_AA)
                #         cv2.imshow('rec', frame)

                # if ((abs(w - h) < 10) and (w < 30 and h < 30) and (x < 230 and y < 350)): #ROI 的範圍（一個矩形）
                #     if frame_count > 220:
                #         cv2.drawContours(frame, c, -1,(0,255,0),3)
                #         cv2.rectangle(frame,(0,0),(230,350),(0,0,255),3,cv2.LINE_AA)
                #         cv2.imshow('rec', frame)
                #         print(lastCenter,'veloCount:', veloCount,' cx:', cx,' cy:', cy,' frame_count:' ,frame_count)
                #     if cx == 0: # 球到ROI框最左邊的時候
                #         lastCenter = (cx,cy)
                #         veloCount += 1
                #         last_timestamp = int(timestamp[frame_count])
                #
                #     else: #球在ROI裡面的時候
                #         veloCount += 1
                #         diffx = abs(cx - lastCenter[0])  #兩顆球之間的距離而已
                #         diffy = abs(cy - lastCenter[1])  #兩顆球之間的距離而已
                #         dist = math.sqrt(diffx**2+diffy**2)  #三角形邊長公式
                #
                #         frame_time = int(timestamp[frame_count]) - int(last_timestamp) #兩顆球timestamp的差別
                #         if frame_time != 0:
                #             frame_rate = 1/(int(timestamp[frame_count]) - int(last_timestamp))*1e9 #變成每秒幾偵
                #         else:
                #             frame_rate = 0
                #
                #         velo = 3600*frame_rate*dist/(1000*pixelToMeter)
                #         print('frame_rate:',frame_rate)
                #
                #         last_timestamp = int(timestamp[frame_count])
                #
                #         if velo > 90 and velo < 160:
                #             tmpVelo.append(velo)
                #
                #         print("veloCount, velo",veloCount, velo)
                #         lastCenter = (cx,cy)
                #         if len(tmpVelo) >= 2:
                #             totalVelo = 0
                #             for oneVelo in tmpVelo:
                #                 totalVelo += oneVelo
                #             pitchVelo = totalVelo/len(tmpVelo)
                #             print("veloCount 3, velo",  pitchVelo, velo)
                #
                #     #writer.writerow([frame_count, cx, cy, dist, frame_rate, velo])
                #     cv2.circle(frame, (int(cx), int(cy)), int(radius), (0,255,0), 2)
                #     #cv2.imshow('frame', frame)
                if len(tmpVelo) == 1:
                    pitchVelo = tmpVelo[0]
                if pitchVelo > 0:
                    cv2.putText(frame, str(pitchVelo), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                                cv2.LINE_AA)
                    # cv2.imshow('frame', frame)
                out.write(frame)
                if cv2.waitKey(0) == 27:
                    cv2.destroyAllWindows()
                    break

            except cv2.error as e:
                break

    print('frame_record: ', frame_record)
    print('pitchVelo_Array', tmpVelo)
    print('pitchVelo: ', pitchVelo)
    cap.release()

    return pitchVelo

from dataclasses import dataclass
from typing import Any
@dataclass
class FrameBlob:
    id: int
    frame: Any
    key_pts: Any
    opening: Any

    def __iter__(self):
        return iter((self.id, self.frame, self.key_pts, self.opening))


def blob2(video_name, frame_path, start_frame=101):
    good_frames = []
    with Timer(name="prepare cv", logger=None):
          # 背景前景分離
        mog.clear()
          # 會返回指定形狀和尺寸的結構元素
        cap = cv2.VideoCapture(video_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with Timer(name="prepare", logger=None):
        fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        minutes = int(duration / 60)
        seconds = duration % 60
        print(f'{frame_count=}, {fps=}, {duration=}')
        print(f'duration (M:S) = {minutes}:{seconds}')

        frame_record_counter = 0
        first_roi_pitch = 100  # check frame from 100

    frame_count = start_frame

    with Timer(name="main", logger=None):
        wait_for_saving_prev_frame_to_array_flag = False
        already_save_prev_frame_to_array_flag = False
        while frame_count - first_roi_pitch <= 20 or frame_record_counter == 0:
            with Timer(name="iterate", logger=None):
                with Timer(name="cap.read", logger=None):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = calibrate_frame_1080p(frame)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                with Timer(name="fgmask", logger=None):
                    fgmask = mog.apply(frame)
                    blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
                    th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
                    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)

                # blank = np.zeros((1, 1))
                # blobs = cv2.drawKeypoints(opening, keypoints, blank, (0, 0, 255),  ###check result
                #                           cv2.DRAW_MATCHES_FLAGS_DEFAULT)

                with Timer(name="round_detect", logger=None):
                    key_points = round_detect(opening)
                with Timer(name="round_detect loop", logger=None):
                    for key_point in key_points:
                        cx, cy = int(key_point.pt[0]), int(key_point.pt[1])  # x,y坐標
                        if 960 < cx < 1920 and 0 < cy < 700:
                            assert prev_frame is not None, "我說不可能"
                            if wait_for_saving_prev_frame_to_array_flag and prev_frame_count != frame_count and not already_save_prev_frame_to_array_flag:
                                print(f'save prev {prev_frame_count=}')
                                good_frames.append(
                                    FrameBlob(
                                        id=prev_frame_count,
                                        frame=prev_frame,
                                        key_pts=key_points,
                                        opening=opening
                                    )
                                )
                                wait_for_saving_prev_frame_to_array_flag = False
                                already_save_prev_frame_to_array_flag = True
                            # frame_record.append(frame_count)  # 記錄可使用的frame
                            # show_blob(blobs,x)  #show blob result

                            if frame_record_counter == 0:
                                first_roi_pitch = frame_count
                                print('first_roi_pitch: ', first_roi_pitch)
                            frame_record_counter = frame_record_counter + 1
                            print(f'save this {frame_count=}')
                            good_frames.append(
                                FrameBlob(
                                    id=frame_count,
                                    frame=frame,
                                    key_pts=key_points,
                                    opening=opening
                                )
                            )
                            break
                        else:
                            prev_frame = frame
                            prev_frame_count = frame_count
                            wait_for_saving_prev_frame_to_array_flag = True

                frame_count += 1
        else:
            cap.release()

    return good_frames


def calc_ball_speed(frames):
    print([f.id for f in frames])
    first_frame_idx = frames[0].id
    tmpVelo = []
    pixelToMeter = 394.44
    lastCenter = (0, 0)
    veloCount = 0
    pitchVelo = 0
    frame_record_counter = 0
    prev_idx = 0

    # print(avis) #check path
    for idx, frame, key_pts, opening in frames:

            blobs = cv2.drawKeypoints(opening, key_pts, np.zeros((1, 1)), (0, 0, 255),  ###check result
                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            for key_point in key_pts:
                cx, cy = int(key_point.pt[0]), int(key_point.pt[1])  # x,y坐標
                # print(f'{idx=}')
                # print(f'{960 < cx < 1920 and 0 < cy < 700=}')
                if 960 < cx < 1920 and 0 < cy < 700:
                    cv2.rectangle(frame, (960, 0), (1920, 700), (0, 0, 255), 3)  # draw roi
                    cv2.circle(blobs, (cx, cy), radius=1, color=(0, 0, 255),
                               thickness=-1)  # draw ball
                    # show_blob(blobs,x)  #show blob result
                    veloCount += 1

                    if frame_record_counter == 0:
                        lastCenter = (cx, cy)
                        frame_record_counter = frame_record_counter + 1
                        prev_idx = first_frame_idx
                        # print('last_frame: ' ,last_frame)

                    else:

                        if idx - prev_idx == 1:
                            frame_record_counter = frame_record_counter + 1
                            # print("current_frame_if: ", current_frame)
                            diffx = abs(cx - lastCenter[0])  # 兩顆球之間的距離而已
                            diffy = abs(cy - lastCenter[1])  # 兩顆球之間的距離而已
                            dist = math.sqrt(diffx ** 2 + diffy ** 2)  # 三角形邊長公式
                            velo = 3600 * (220) * dist / (1000 * pixelToMeter)
                            # print("inside velo:",velo)
                            if velo > 60 and velo < 160:
                                tmpVelo.append(velo)
                            # print("Last Current: ", last_frame,current_frame)
                        prev_idx = idx

                        lastCenter = (cx, cy)
                    if len(tmpVelo) >= 2:
                        totalVelo = 0
                        for oneVelo in tmpVelo:
                            totalVelo += oneVelo
                        pitchVelo = totalVelo / len(tmpVelo)

    print('pitchVelo_Array', tmpVelo)
    print('pitchVelo: ', pitchVelo)


# aviFiles = glob.glob(os.path.join("output", "*.MOV"))  #go all file

# emptydir('outputMP4')
# emptydir('data')
# video_info = []


# with open(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\ballspeed_log.csv','w') as t:
#     t.write('')
#     for avi in aviFiles:
#         time_start = time.time()
#
#         # print(avi)
#         video_name = avi.split('\\')[1]
#         # print(video_name)
#         # video_info = video_name.split('')
#         # print(video_info)
#         velo = blob(avi, 'outputMP4', video_name)
#         time_end = time.time()
#         print('processing time:', time_end - time_start, 's')
#         # velo = blob(avi,'outputMP4', video_info)
#         with open(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\ballspeed_log.csv','a') as f:
#             f.write(str(velo)+','+video_name)
#             f.write('\n')

# mtx = [[4600.98769128, 0, 961.17445224], [0, 4574.72596027, 537.80462204], [0, 0, 1]]
# dist = [[0.84660277, -15.05073586, 0.06927329, 0.04566403, 105.27604409]]
# mtx = np.asarray(mtx)
# dist = np.asarray(dist)
# undistortion_check(mtx, dist,r"C:\Users\Ricky\PycharmProjects\server\file\cal_video\trim.D8847D3F-F4F2-4A07-82DC-3DFAB44ECA3A.MOV")


# ball_speed = blob(r"C:\Users\Ricky\PycharmProjects\server\file\cal_video\trim.D8847D3F-F4F2-4A07-82DC-3DFAB44ECA3A.MOV",'outputMP4')  # 337 frames

if __name__ == '__main__':
    start_time = time.perf_counter()
    good_frames = blob2(
        video_name=r"C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\source_video\output_20221109164853.MOV",
        frame_path="C:/Users/Ricky/PycharmProjects/baseball/BallSpeedCode/frame_save"
    )
    mid_time = time.perf_counter()
    print('blob processing time', mid_time - start_time, 's')

    pd.options.display.max_columns = None
    pd.options.display.max_colwidth = None
    timer_df = pd.DataFrame()
    for timer_name in Timer.timers:
        timer_df = pd.concat([timer_df, pd.DataFrame({
            'timer': timer_name,
            'count': Timer.timers.count(timer_name),
            'total': Timer.timers.total(timer_name),
            'min': Timer.timers.min(timer_name),
            'max': Timer.timers.max(timer_name),
            'mean': Timer.timers.mean(timer_name),
            'median': Timer.timers.median(timer_name),
            'stdev': Timer.timers.stdev(timer_name),
        }, index=[0])], ignore_index=True)

    print(timer_df)
    mtx = np.array([[4600.98769128, 0, 961.17445224], [0, 4574.72596027, 537.80462204], [0, 0, 1]])
    dist = np.array([[0.84660277, -15.05073586, 0.06927329, 0.04566403, 105.27604409]])
    mtx = np.asarray(mtx)
    dist = np.asarray(dist)
    fisheye_start = time.time()
    undistortion_V2(mtx, dist)
    fisheye_end = time.time()
    print('fisheye processing time', fisheye_end - fisheye_start, 's')
#
    calc_ball_speed(good_frames)
    # blob2(r"C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\source_video\output_20221109164853.MOV",'outputMP4')
    end_time = time.time()
    print('cal processing time', end_time - fisheye_end, 's')
