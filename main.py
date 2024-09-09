#!/usr/bin/env python3

import cv2 as cv
import numpy as np

def detect_circles(gray, min_radius=10, max_radius=30):
    return cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 2, rows / 8,
                           param1=100, param2=0.7,
                           minRadius=min_radius, maxRadius=max_radius)

# algorithm: color and circle detection
# pupil is always black,
# and sclera (part around iris) is always white
# these seems like a great detectors for eyes
# - iris is easy to detect as a circle, pupil not so
#   easy because of reflections

def put_text(img, pos, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,text,pos, font, 1,(255,255,255),1,cv.LINE_AA)

def avg(a, b):
    return (a + b)/2

def try_hsl(frame, logger):
    hsl = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    mask = cv.inRange(hsl[:,:,0], 90, 120)

    # Morphology on mask
    kernel = np.ones((13,13), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.dilate(mask, kernel, iterations = 4)
    #cv.imshow("mask", mask)

    #cv.imshow('f', hsl[:,:,0])
    #cv.imshow('fm', mask)

    # Find contours and select the once matching eyes
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        # w/h needs to be in some range
        if not (w/h > 0.5 and w/h < 2.5):
            continue

        # Now try detecting circles
        roi = frame[y:y+h, x:x+w]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        circles = detect_circles(gray, min_radius = int(h/2 * 0.6), max_radius = int(h/2 * 0.95))
        if circles is None:
            continue

        # Only plot rectangle when there are circles
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Plot iris
        for [nx, ny, nr] in circles[0, :].astype(int):
            ix = x + nx
            iy = y + ny
            cv.circle(frame, (ix, iy), nr, (0, 0, 255), 2)

            # Find a rectangle inside iris that should contain pupil
            pr = int(nr*0.66)
            px = ix - pr
            py = iy - pr
            cv.rectangle(frame, (px, py), (px + 2*pr, py + 2*pr), (0,0,255), 2)

            # Find pupil using color select
            roi = frame[py:py+2*pr, px:px+2*pr]
            if roi.size == 0:
                continue
            hsl = cv.cvtColor(roi, cv.COLOR_BGR2HLS)
            mask = cv.inRange(hsl[:,:,1], 0, 25)

            # Morphology on mask
            kernel = np.ones((3,5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.dilate(mask, kernel, iterations = 3)

            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            minp = [px + pr, py + pr]
            maxp = [px + pr, py + pr]
            # Combine contours into a single feature
            for contour in contours:
                cx,cy,cw,ch = cv.boundingRect(contour)
                gx = px + cx
                gy = py + cy

                minp[0] = min(minp[0], gx, gx + cw)
                minp[1] = min(minp[1], gy, gy + ch)

                maxp[0] = max(maxp[0], gx, gx + cw)
                maxp[1] = max(maxp[1], gy, gy + ch)
                #cv.rectangle(frame, (gx, gy), (gx+cw, gy+ch), (0, 255, 0), 2)

            # Compute pupil radius estiamte from this
            pr = min(maxp[0] - minp[0], maxp[1] - minp[1])
            pr = int(pr/2)

            #["Eye size", "Iris size", "Pupil size", "Pupil/Iris"]
            #put_text(frame, (x, y), str(pr/nr))
            if logger.ready():
                smooth_ratio = str(logger.get()[-1])
                ratio_smooth = str(logger.get()[-2]/logger.get()[-3])
                #put_text(frame, (x, y), smooth_ratio)
                put_text(frame, (x, y), ratio_smooth)
                print(ratio_smooth)
                #print(f"{smooth_ratio}\t{ratio_smooth}")

            if pr > 0:
                logger.log([avg(w, h), nr, pr, pr/nr])
                #print(pr)
                cv.circle(frame, (ix, iy), pr, (255, 0, 0), 2)
                #print(minp, maxp)
                #cv.rectangle(frame, minp, maxp, (255, 0, 0), 2)
            #cv.imshow("mask", mask)
            #cv.imshow("hsl", hsl)

    #res = cv.bitwise_and(frame, frame, mask = mask)


    logger.plot(frame)
    cv.imshow('frame', frame)
    #cv.imshow("hsl", hsl)
    #cv.imshow('mask', mask)
    #cv.imshow('res', res)

class NoiseReducer:
    def __init__(self):
        from collections import deque
        self.buf = deque(maxlen = 20)

    def append(self, data):
        self.buf.append(data)

    def ready(self):
        return len(self.buf) == self.buf.maxlen

    def get(self):
        import numpy as np
        result = np.zeros(len(self.buf[0]))
        for v in self.buf:
            result += np.array(v)
        result /= len(self.buf)

        return result

class Logger:
    def __init__(self, path, smooth_path, columns):
        self.path = path
        self.smooth_path = smooth_path
        self.columns = columns
        self.data_idx = 0

        # Data buffers for plotting
        from collections import deque
        self.noise_reducer = NoiseReducer()
        self.data_buf = deque(maxlen = 2000)
        self.data_buf_times = deque(maxlen = self.data_buf.maxlen)
        self.data_buf_idx = 0

    def log(self, row):
        from datetime import datetime
        if len(row) != len(self.columns):
            print("Logger Error: Number of entries in a row doesn't match number of columns")
            return

        def write_row(data_file, data_idx, row):
            data_file.write(str(data_idx))
            data_file.write(f",{datetime.now()}")
            for el in row:
                data_file.write(f",{el}")
            data_file.write('\n')

        write_row(self.data_file, self.data_idx, row)
        self.data_idx += 1

        # Plotting
        self.noise_reducer.append(row)
        if self.noise_reducer.ready():
            buf = self.noise_reducer.get().tolist()
            self.data_buf.append(buf)
            write_row(self.smooth_data_file, self.data_buf_idx, buf)
            self.data_buf_times.append(datetime.now())
            self.data_buf_idx += 1 # TODO: rename to smooth_data_*
            #print(buf)

    def plot(self, img):
        if not self.ready():
            return

        (h, w, _) = img.shape

        dy = h
        dx = w/((self.data_buf_times[-1] - self.data_buf_times[0]).total_seconds() + 1)

        y0 = self.data_buf[0][-1]
        x0 = self.data_buf_times[0]

        p1 = (0, 0)
        for i in range(1, len(self.data_buf)):
            y1 = self.data_buf[i][-1]
            x1 = self.data_buf_times[i]

            p2 = (int((x1 - x0).total_seconds()*dx), int((1-y1)*dy))

            cv.line(img, p1, p2, (0, 0, 255), 2)
            p1 = p2

    # Get the latest measurement
    def ready(self):
        return not not self.data_buf

    def get(self):
        return self.data_buf[-1]

    def __enter__(self):
        self.data_file = open(self.path, 'w')
        self.smooth_data_file = open(self.smooth_path, 'w')

        def write_cols(data_file):
            data_file.write(",Timestamp") # add index and timestamp automatically
            for column in self.columns:
                data_file.write(f",{column}")
            data_file.write('\n')

        write_cols(self.data_file)
        write_cols(self.smooth_data_file)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.data_file.close()
        self.smooth_data_file.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--video-in', default=0)
    parser.add_argument('-l', '--log-out', default="log.csv")
    parser.add_argument('-s', '--smooth-log-out', default="slog.csv")
    parser.add_argument('-o', '--video-out')
    args = parser.parse_args()
    if args.video_out:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        vout = cv.VideoWriter(args.video_out, fourcc, 20.0, (640, 480))

    video = cv.VideoCapture(args.video_in)

    video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    columns = ["Eye size", "Iris size", "Pupil size", "Pupil/Iris"]
    with Logger(args.log_out, args.smooth_log_out, columns) as logger:
        paused = False
        while True:
            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
            if keyboard == 'p':
                paused = True
            if keyboard == 'r':
                paused = False

            if paused:
                continue

            ret, frame = video.read()
            if not ret:
                break

            if args.video_out:
                vout.write(frame)
            rows = frame.shape[0]
            try_hsl(frame, logger)

    video.release()
    if args.video_out:
        vout.release()
