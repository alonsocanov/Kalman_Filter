import cv2
import numpy as np
import sys
import os
import glob
import imutils
import argparse
from kalman_filter import KalmanFilter as KF


def availableFiles(path) -> list:
    full_path = glob.glob(path, recursive=True)
    if len(full_path):
        file = full_path[0]
    else:
        error = ' '.join(['Could not find file:', path])
        sys.exit(error)
    return file


def checkWebcamAvalability(webcam: cv2.VideoCapture) -> None:
    if not webcam.isOpened():
        sys.exit("Error opening webcam")


def check_Q() -> bool:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


# color normalization of HSV to OpenCV HSV
def hsv2cvhsv(hsv: np.array) -> np.array:
    # For HSV, Hue range is [0,179], Saturation range is [0,255]
    # and Value range is [0,255]. Different software use different scales.
    # So if you are comparinn in OpenCV values with them, you need to normalize these ranges.
    hsv_cv = np.array([179, 255, 255])
    hsv_orig = np.array([360, 100, 100])
    cv_hsv = np.divide((hsv * hsv_cv), hsv_orig)
    return cv_hsv


def drawTrace(img: np.array, pts: np.array) -> None:
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            pass
        else:
            thickness = int(np.sqrt(100 / float(i + 1)) * 2.5)
            cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), thickness)


def main():
    parser = argparse.ArgumentParser(description='Ball detection and Kalman Filter')
    parser.add_argument('--video', type=str, help='Video path to do some tracking')
    args = parser.parse_args()
    if args.video:
        file = availableFiles(args.video)
        if file:
            video = cv2.VideoCapture(file)
        else:
            video.release()
            error = ' '.join(['Could not find file:', path])
            sys.exit(error)
    else:
        try:
            video = cv2.VideoCapture(0)
            print('Using webcam')
        except:
            video.release()
            sys.exit('There isn\'t a webcam')

    checkWebcamAvalability(video)
    ret, frame = video.read()
    h, w = frame.shape[:2]

    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    hsv_lower = hsv2cvhsv(np.array([45, 40, 20]))
    hsv_upper = hsv2cvhsv(np.array([65, 100, 100]))

    acce = np.array([20, .5]).reshape(-1, 1)
    acce_var = np.array([20, 20, 20, 20]).reshape(-1, 1)
    meas_var = np.array([.01, .01, .01, .01]).reshape(-1, 1)
    pts = list()

    flag_obj = 0
    print('Press Q to quit')
    t = 0
    fps = 24
    delta_t = fps / 60
    while video.isOpened():
        ret, frame = video.read()
        if check_Q() or frame is None:
            break
        frame = cv2.resize(frame, (w // 2, h // 2))
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        i = 0
        if len(cnts) > 0:
            try:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if r > 10:
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 4)
                    cv2.circle(frame, center, 1, (0, 0, 255), 5)
                    cv2.putText(frame, 'z', (int(x) + 5, int(y) + 5), font, fontScale, fontColor, lineType)
                    z = np.array([x, y]).reshape(-1, 1)
                    if not flag_obj:
                        x_0 = z
                        flag_obj = 1
                    elif flag_obj == 1:
                        v_0 = abs(z - x_0) / delta_t
                        x_0 = z
                        kf = KF(x_0, v_0, acce)
                        flag_obj = 2
                    else:
                        v = abs(z - x_0) / delta_t
                        x_0 = z
                        meas = np.append(z, v).reshape(-1, 1)
                        kf.update(meas, meas_var)
                        kf.predict(delta_t, acce_var)

                        i += 1
                elif flag_obj == 2:
                    kf.predict(delta_t, acce_var)

                # pts.insert(0, center)
                # if len(pts) > 100:
                #     pts.pop()

                # drawTrace(frame, pts)
                if flag_obj == 2:
                    x, y = kf.pos.astype(int)
                    cv2.circle(frame, (x, y), 1, (255, 0, 255), 5)
                    cv2.putText(frame, 'KF', (x - 5, y - 5), font, fontScale, fontColor, lineType)

            except:
                pass

        t += delta_t

        img = frame

        cv2.imshow(win_name, img)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
