import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path


class NMOT():
    def __init__(self, knn_history=100, dist2Threshold=50.0, detectShadows=False, lifespan=100, goodfeatures2track = True):
        
        
        # knn params
        self.knn_history = knn_history
        self.dist2Threshold = dist2Threshold
        self.detectShadows = detectShadows
        self.knn_subtractor = cv.createBackgroundSubtractorKNN(history = self.knn_history,
                                                               dist2Threshold = self.dist2Threshold,
                                                               detectShadows = self.detectShadows)
        
        # # alternative MOG subtractor
        # self.mog_history = 200
        # self.mog_varThreshold = 100
        # self.mog_detectShadows = False
        # self.mog_subtractor = cv.createBackgroundSubtractorMOG2(history=self.mog_history, varThreshold=self.mog_varThreshold, detectShadows=self.mog_detectShadows)

        self.lifespan = lifespan

        
        # lk params
        self.lk_params = dict( winSize  = (21, 21),
                               maxLevel = 1,
                               criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                               )

        # blur params
        self.gaussian_ksize = (7,7)
        self.gaussian_sigma = None

        # morphology filter params
        self.morph_kernel = cv.getStructuringElement(cv.MORPH_DIAMOND, (3,3))
        self.morph_iterations = 1

        self.warmup_frames = 50
        self._warmup_left = self.warmup_frames
        self.old_gray = None
        self.gf2t = goodfeatures2track

    def update(self, frame):
        
        frame = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # if self._warmup_left==0:
            
        #     self._new_points = self._optical_flow(self.old_gray, frame_gray, self._old_points)
        #     frame_with_markers = self._draw_markers(frame, self._new_points)
        # else:    
        blured = cv.GaussianBlur(src = frame_gray, ksize = self.gaussian_ksize, sigmaX = self.gaussian_sigma)
        
        self.knn_mask = self.knn_subtractor.apply(blured)
        self.morphology_result = self._morphology(self.knn_mask)

    
        self.contours, self.hierarchy = cv.findContours(self.morphology_result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self._old_points = self._find_centers(self.contours)

        if len(self._old_points) > 0:
            frame_with_markers = self._draw_markers(frame, self._old_points)
        else:
            frame_with_markers = frame

        # if self._warmup_left > 0:
        #     self._warmup_left -= 1
        
        self.old_gray = frame_gray.copy()
        
        return frame_with_markers, self.morphology_result


    
    def _morphology(self, frame):
        frame = frame.copy()
        self.close = cv.morphologyEx(frame, op = cv.MORPH_CLOSE, kernel = self.morph_kernel, iterations = self.morph_iterations)
        self.open = cv.morphologyEx(self.close, op = cv.MORPH_OPEN, kernel = self.morph_kernel, iterations = self.morph_iterations)

        return self.open
    
    def _find_centers(self, contours):
        centers = []
        for contour in contours:
            # find center of each contour
            M = cv.moments(contour)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            centers.append((center_X, center_Y))

        return np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
    
    def _draw_markers(self, frame, points):
        frame = frame.copy()
        for x, y in points.squeeze(1):
            cv.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
        return frame
    
    def _draw_contours(self, frame):
        frame = frame.copy()
        frame = cv.drawContours(frame, self.contours, -1, (0, 255, 0), 1)
        
        return frame
    
    def _optical_flow(self, old_gray, frame_gray, old_points):
        
        if old_points is None or len(old_points) == 0:
            return None
        
        
        self._new_points, self._st, self._err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, old_points, None, **self.lk_params
        )
        
        if self._new_points is not None and self._st is not None:
            good_new = self._new_points[self._st.flatten() == 1]
            
            return good_new
        else:
            return None
