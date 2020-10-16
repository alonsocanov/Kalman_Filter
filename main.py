from kalman_filter import KalmanFilter as Kf
import numpy as np
import cv2
import argparse
import glob
import imutils
import sys


def resize(img: np.ndarray, rescale_factor: float=None):
    h_max = 600
    h, w = img.shape[:2]
    if not rescale_factor and h_max:
        rescale_factor = h_max / h
    dimesion = (int(w * rescale_factor), int(h * rescale_factor))
    resized_img = cv2.resize(img, dimesion)
    return resized_img, 1 / rescale_factor


# check available files in path
def availableFiles(path) -> list:
    full_path = glob.glob(path, recursive=True)
    if len(full_path):
        file = full_path[0]
    else:
        file = None
    return file


# check if webcam is already opened
def checkWebcamAvalability(webcam: cv2.VideoCapture) -> None:
    if not webcam.isOpened():
        sys.exit("Error opening webcam")


# check if key q was pressed
def check(c: str='q') -> bool:
    if cv2.waitKey(1) & 0xFF == ord(c):
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


# draw object tail with image and points coordenates
def drawTrace(img: np.array, pts: np.array, color: tuple=(0, 0, 255), line_width:int=2) -> None:
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            pass
        else:
            cv2.line(img, pts[i - 1], pts[i], color, line_width)


def main():
    parser = argparse.ArgumentParser(description='Ball detection and Kalman Filter')
    parser.add_argument('--path', type=str, default='data/ball_straight.mp4', help='Video path to do some tracking')
    parser.add_argument('--save', type=bool, default=False, help='Save video output')
    args = parser.parse_args()
    if args.path:
        file = availableFiles(args.path)
        if file:
            video = cv2.VideoCapture(file)
        else:
            video.release()
            error = ' '.join(['Could not find file:', args.path])
            sys.exit(error)

    flag_first_detect = False
    flag_second_detect = False

    ret, frame = video.read()
    frame, factor = resize(frame)
    # height and width
    h, w = frame.shape[:2]

    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    fontColor = (0, 255, 255)
    lineType = 2

    # lower and upper range of hsv color
    hsv_lower = hsv2cvhsv(np.array([10, 40, 0]))
    hsv_upper = hsv2cvhsv(np.array([50, 100, 100]))

    pts = list()
    kf_pos = list()
    if args.save:
        out = cv2.VideoWriter('output/out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (w, h))

    x_prev = []
    v_prev = []
    var = 20
    meas_var = 10
    delta_t = 100 / 24
    kf = None
   

    print('Press Q to quit')
    while video.isOpened():
        ret, frame = video.read()
        if check() or frame is None:
            break
        frame, factor = resize(frame)
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            except ZeroDivisionError:
                center = False
            if r > 25 and center:
                if not flag_first_detect:
                    flag_first_detect = True
                elif flag_first_detect and not flag_second_detect:
                    x_i = np.array([x, y])
                    v_i = (x_i - x_prev) / delta_t
                    kf = Kf(x_i, v_i, 10)
                    flag_second_detect = True
                elif flag_first_detect and flag_second_detect:
                    x_i = np.array([x, y])
                    v_i = (x_i - x_prev) / delta_t
                    z_i = np.concatenate((x_i, v_i), axis=0)
                    kf.update(z_i, meas_var)
                    kf.predict(delta_t, [0, 0], var)
                x_prev = np.array([x, y])

                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 4)
                cv2.circle(frame, center, 1, (0, 0, 255), 5)
                cv2.putText(frame, 'z', (int(x) + 10, int(y) + 10), font, fontScale, fontColor, lineType)
                pts.insert(0, center)
                drawTrace(frame, pts)

        elif flag_first_detect and flag_second_detect:
            kf.predict(delta_t, [0, 0], var)

        if kf:
            x_kf, y_kf = kf.pos
            cv2.circle(frame, (int(x_kf), int(y_kf)), 35, (255, 0, 0), 4)
            cv2.putText(frame, 'kf', (int(x_kf) - 10, int(y_kf) + 10), font, fontScale, fontColor, lineType)
            kf_pos.insert(0, (int(x_kf), int(y_kf)))
            drawTrace(frame, kf_pos, (255, 255, 0), 1)


        # save video feed
        if args.save:
            out.write(frame)
        img = frame
        cv2.imshow(win_name, img)
        # cv2.waitKey(1000)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
