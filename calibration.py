
import numpy as np
import cv2
from cv2 import VideoCapture
import glob
import os
import shutil

def undistortion(mtx, dist):
    #picture
    # img = cv2.imread(r'source_video\messageImage_1655541173218.jpg')
    img = cv2.imread(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\frame_save\frame178.jpg')

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("newmtx:", newcameramtx)
    print("roi:", roi)

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    dst = cv2.resize(dst, (1920, 1080))
    # cv2.imwrite('output/messageImage_1655541173218.jpg', dst)
    cv2.imwrite(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\calibration_save\frame178.jpg', dst)

    #video
    # avis = glob.glob('source_video/*.mov')
    # for avi in avis:
    #     cap = VideoCapture(avi)
    #     ret = True
    #     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #     out_name = avi.split('\\')[1]
    #     print(out_name)
    #     out = cv2.VideoWriter('output/' + out_name, fourcc, 30.0, (1920,1080))
    #
    #     while ret:
    #         ret, frame = cap.read()
    #         # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #
    #         if ret:
    #             try:
    #                 h, w = frame.shape[:2]
    #                 newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    #                 dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    #                 x,y,w,h = roi
    #                 dst = dst[y:y+h, x:x+w]
    #                 dst = cv2.resize(dst, (1920, 1080))
    #                 out.write(dst)
    #                 #cv2.imshow('frame', dst)
    #                 if cv2.waitKey(0) == 27:
    #                     cv2.destroyAllWindows()
    #                     break
    #             except cv2.error as e:
    #                 break
            
        

def undistortion_check(mtx, dist,video_path):
    ### picture
    # img = cv2.imread('source_video\messageImage_1655541173218.jpg')
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # print("newmtx:", newcameramtx)
    # print("roi:", roi)

    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # dst = cv2.resize(dst, (720, 540))
    # cv2.imwrite('output/messageImage_1655541173218.jpg', dst)

    ### video
    # avis = glob.glob('source_video/*.mov')
    avi = video_path
    # for avi in avis:

    cap = VideoCapture(avi)
    ret = True
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_name = avi.split('\\')[1]
    print(out_name)
    out = cv2.VideoWriter('file/cal_video/' + out_name, fourcc, 30.0, (1920, 1080))

    while ret:
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if ret:
            try:
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                dst = cv2.resize(dst, (1920, 1080))
                out.write(dst)
                # cv2.imshow('frame', dst)
                if cv2.waitKey(0) == 27:
                    cv2.destroyAllWindows()
                    break
            except cv2.error as e:
                break


def undistortion_V2(mtx, dist):
    ### picture
    # avis = glob.glob('./BallSpeedCode/frame_save/*.jpg')
    avis = glob.glob(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\frame_save\*.bmp')
    print(avis)
    for avi in avis:
        picName = avi.split("\\")[-1]
        # print(picName)  #check name
        img = cv2.imread(avi)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # print("newmtx:", newcameramtx)
        # print("roi:", roi)

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (1920, 1080))
        # cv2.imwrite('output/messageImage_1655541173218.jpg', dst)
        cv2.imwrite(r'C:\Users\Ricky\PycharmProjects\baseball\BallSpeedCode\calibration_save\\' + picName , dst)

cam_mtx = np.array([[4600.98769128, 0, 961.17445224], [0, 4574.72596027, 537.80462204], [0, 0, 1]])
cam_dist = np.array([[0.84660277, -15.05073586, 0.06927329, 0.04566403, 105.27604409]])
def calibrate_frame_1080p(img):
    frame_shape = 1920, 1080

    img_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, cam_dist, frame_shape, 1, frame_shape)
    dst = cv2.undistort(img, cam_mtx, cam_dist, None, img_mtx)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return cv2.resize(dst, frame_shape)

### video
# # avis = glob.glob('source_video/*.mov')
# avi = video_path
# # for avi in avis:
#
# cap = VideoCapture(avi)
# ret = True
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out_name = avi.split('\\')[1]
# print(out_name)
# out = cv2.VideoWriter('file/cal_video/' + out_name, fourcc, 30.0, (1920, 1080))
#
# while ret:
#     ret, frame = cap.read()
#     # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     if ret:
#         try:
#             h, w = frame.shape[:2]
#             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#             dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#             x, y, w, h = roi
#             dst = dst[y:y + h, x:x + w]
#             dst = cv2.resize(dst, (1920, 1080))
#             out.write(dst)
#             # cv2.imshow('frame', dst)
#             if cv2.waitKey(0) == 27:
#                 cv2.destroyAllWindows()
#                 break
#         except cv2.error as e:
#             break

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        
    os.mkdir(dirname)  

# emptydir('output')

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((4*4, 3), np.float32)
# objp[:,:2] = np.mgrid[0:4, 0:4].T.reshape(-1,2)
#
# objpoints = []
# imgpoints = []
#
#
# images = glob.glob('old_chessboard/*.jpg')
#
# for fname in images:
#     img = cv2.imread(fname)
#     img = cv2.resize(img, (1920, 1080))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, corners = cv2.findChessboardCorners(gray, (4,4), None)
#
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#         imgpoints.append(corners)
#         cv2.drawChessboardCorners(img, (4,4), corners2, ret)
#     # cv2.imshow('frame', img)
#     if cv2.waitKey(0) == 27:
#         break
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# # np.set_printoptions(suppress=True)
# print("mtx:", mtx)
# print("dist:", dist)
# print(type(mtx),type(dist))

# mtx = [ [4600.98769128,0,961.17445224],[0,4574.72596027,537.80462204],[0,0,1]]
# dist = [[0.84660277 ,-15.05073586  , 0.06927329 ,  0.04566403, 105.27604409]]
# mtx = np.asarray(mtx)
# dist = np.asarray(dist)
# print(type(mtx),type(dist))

# undistortion(mtx, dist)

# mtx = [ [4600.98769128,0,961.17445224],[0,4574.72596027,537.80462204],[0,0,1]]
# dist = [[0.84660277 ,-15.05073586  , 0.06927329 ,  0.04566403, 105.27604409]]
# mtx = np.asarray(mtx)
# dist = np.asarray(dist)
# undistortion_V2(mtx,dist)
# undistortion(mtx,dist)


