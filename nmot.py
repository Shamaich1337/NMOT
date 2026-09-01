import cv2 as cv
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from scipy.optimize import linear_sum_assignment

@dataclass
class Track:
    track_id: int
    pos: np.ndarray
    prev_pos: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    age: int = 0
    hits: int = 1
    missed: int = 0
    history: List[Tuple[int, float, float, str]] = field(default_factory=list)


class NMOT:
    
    def __init__(
        self,
        knn_history: int = 100,
        dist2Threshold: float = 50.0,
        detectShadows: bool = False,
        warmup_frames: int = 50,
        min_area: float = 8.0,
        max_area: float = 1000.0,
        max_match_dist: float = 25.0,
        max_missed: int = 10,
        # use_lk: bool = True,
        pred_head: Optional[Literal['kalman', 'lucas-kanade']] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.knn_subtractor = cv.createBackgroundSubtractorKNN(
            history=knn_history,
            dist2Threshold=dist2Threshold,
            detectShadows=detectShadows,
        )

        self.warmup_frames = warmup_frames
        self.min_area = min_area
        self.max_area = max_area
        self.max_match_dist = max_match_dist
        self.max_missed = max_missed
        # self.use_lk = use_lk
        self.pred_head = pred_head
        self.roi = roi

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        valid_pred_heads = (None, 'kalman', 'lucas-kanade')
        if pred_head not in valid_pred_heads:
            raise ValueError(
                f"pred_head must be one of {valid_pred_heads}, got {pred_head}"
            )
        
        if self.pred_head=='kalman':
            self.kalman_filters: Dict[int, cv.KalmanFilter] = {}

        self.gaussian_ksize = (3, 3)
        self.morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        self.morph_iterations = 1

        self.frame_idx = 0
        self.prev_gray = None

        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.rows = []

        self.last_mask = None
        self.last_contours = []
        self.last_detections = np.empty((0, 2), dtype=np.float32)

    def _init_kalman(self, pos):
        # init constant velocity kalman filter
        kf = cv.KalmanFilter(dynamParams = 4, measureParams = 2)
        # [x, y, dx, dy]; [x, y]

        kf.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]], np.float32)

        kf.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1], 
                                        [0,0,1,0], 
                                        [0,0,0,1]], np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2 
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.statePost = np.array([[pos[0]], [pos[1]], [0], [0]], np.float32)
        return kf
    
    def update(self, frame: np.ndarray):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = self._foreground_mask(frame_gray)

        self.last_mask = mask
        self.last_contours = []
        self.last_detections = np.empty((0, 2), dtype=np.float32)

        if self.frame_idx >= self.warmup_frames:
            # knn + morphology + centroid extraction
            detections, contours = self._detect_centroids(mask)

            self.last_detections = detections
            self.last_contours = contours

            if self.pred_head=='lucas-kanade':
                predictions = self._predict_tracks_lucas_kanade(frame_gray)
            elif self.pred_head=='kalman':
                predictions = self._predict_tracks_kalman()
            elif self.pred_head is None:
                predictions  = {tid: track.pos for tid, track in self.tracks.items()}
            else:
                raise ValueError(f"Unknown pred_head: {self.pred_head}")
            matches, unmatched_tracks, unmatched_detections = self._associate(
                predictions,
                detections
            )

            self._update_matched_tracks(matches, detections, predictions)
            self._update_unmatched_tracks(unmatched_tracks, predictions)
            self._create_new_tracks(unmatched_detections, detections)
            self._remove_dead_tracks()

        vis = self._draw(frame)

        self.prev_gray = frame_gray.copy()
        self.frame_idx += 1

        return vis, mask, list(self.tracks.values())

    def _foreground_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv.GaussianBlur(gray, self.gaussian_ksize, 0)
        mask = self.knn_subtractor.apply(blurred)

        if self.roi is not None:
            x, y, w, h = self.roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y:y + h, x:x + w] = 255
            mask = cv.bitwise_and(mask, roi_mask)

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            self.morph_kernel,
            iterations=self.morph_iterations,
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            self.morph_kernel,
            iterations=self.morph_iterations,
        )

        return mask

    def _detect_centroids(self, mask: np.ndarray):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detections = []
        valid_contours = []

        for contour in contours:
            area = cv.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            moments = cv.moments(contour)

            if abs(moments["m00"]) > 1e-6:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                x, y, w, h = cv.boundingRect(contour)
                cx = x + w / 2
                cy = y + h / 2

            detections.append([cx, cy])
            valid_contours.append(contour)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32), valid_contours

        return np.asarray(detections, dtype=np.float32), valid_contours

    def _predict_tracks_lucas_kanade(self, gray: np.ndarray) -> Dict[int, np.ndarray]:
        predictions = {}

        if len(self.tracks) == 0:
            return predictions

        track_ids = list(self.tracks.keys())

        if self.prev_gray is None:
            return predictions

        old_points = np.asarray(
            [self.tracks[tid].pos for tid in track_ids],
            dtype=np.float32,).reshape(-1, 1, 2)

        new_points, status, err = cv.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            old_points,
            None,
            **self.lk_params,
        )

        if new_points is None or status is None:
            return predictions

        new_points = new_points.reshape(-1, 2)
        status = status.reshape(-1)

        for tid, point, ok in zip(track_ids, new_points, status):
            if int(ok) == 1:
                predictions[tid] = point.astype(np.float32)

        return predictions


    def _predict_tracks_kalman(self):
        predictions = {}

        if len(self.tracks) == 0:
            return predictions

        for tid, tr in self.tracks.items():
            # init new filter for new track
            if tid not in self.kalman_filters:
                self.kalman_filters[tid]=self._init_kalman(tr.pos)
            
            kf = self.kalman_filters[tid]
            
            pred = kf.predict()
            point = pred[:2].flatten()
            predictions[tid] = point
        

        return predictions

    def _associate(self, predictions: Dict[int, np.ndarray], detections: np.ndarray):
        track_ids = list(predictions.keys())

        if len(track_ids) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], track_ids, []

        pred_points = np.asarray(
            [predictions[tid] for tid in track_ids],
            dtype=np.float32,
        )

        diff = pred_points[:, None, :] - detections[None, :, :]
        cost = np.linalg.norm(diff, axis=2)

        matches = []
        used_tracks = set()
        used_detections = set()

        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.max_match_dist:
                tid = track_ids[r]
                matches.append((tid, c))
                used_tracks.add(tid)
                used_detections.add(c)

        unmatched_tracks = [
            tid for tid in track_ids
            if tid not in used_tracks
        ]

        unmatched_detections = [
            i for i in range(len(detections))
            if i not in used_detections
        ]

        return matches, unmatched_tracks, unmatched_detections

    def _update_matched_tracks(self, matches, detections, predictions):
        for tid, det_idx in matches:
            tr = self.tracks[tid]

            new_pos = detections[det_idx].astype(np.float32)

            if self.pred_head=='kalman' and tid in self.kalman_filters:
                kf = self.kalman_filters[tid]
                kf.correct(new_pos.reshape(-1, 1))

            tr.prev_pos = tr.pos.copy()
            tr.pos = new_pos
            tr.velocity = tr.pos - tr.prev_pos
            tr.age += 1
            tr.hits += 1
            tr.missed = 0

            self._record(tr, status="matched")

    def _update_unmatched_tracks(self, unmatched_tracks, predictions):
        for tid in unmatched_tracks:
            tr = self.tracks[tid]
            predicted = predictions[tid].astype(np.float32)
            
            tr.prev_pos = tr.pos.copy()
            tr.pos = predicted
            tr.velocity = tr.pos - tr.prev_pos
            tr.age += 1
            tr.missed += 1

            self._record(tr, status="predicted")

    def _create_new_tracks(self, unmatched_detections, detections):
        for det_idx in unmatched_detections:
            pos = detections[det_idx].astype(np.float32)

            tr = Track(
                track_id=self.next_id,
                pos=pos.copy(),
                prev_pos=pos.copy(),
            )

            self.tracks[self.next_id] = tr
            self._record(tr, status="new")

            self.next_id += 1

    def _remove_dead_tracks(self):
        dead_tracks = [
            tid for tid, tr in self.tracks.items()
            if tr.missed > self.max_missed
        ]

        for tid in dead_tracks:
            if self.pred_head=='kalman':
                self.kalman_filters.pop(tid, None)
            del self.tracks[tid]

    def _record(self, tr: Track, status: str):
        x, y = tr.pos

        tr.history.append(
            (
                self.frame_idx,
                float(x),
                float(y),
                status,
            )
        )

        self.rows.append(
            {
                "frame": self.frame_idx,
                "track_id": tr.track_id,
                "x": float(x),
                "y": float(y),
                "status": status,
                "missed": tr.missed,
            }
        )

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()

        cv.drawContours(vis, self.last_contours, -1, (0, 255, 255), 1)

        for det in self.last_detections:
            x, y = det
            cv.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

        for tid, tr in self.tracks.items():
            x, y = tr.pos
            color = self._color_by_id(tid)

            if tr.missed > 0:
                cv.circle(vis, (int(x), int(y)), 6, color, 1)
            else:
                cv.circle(vis, (int(x), int(y)), 4, color, -1)

            cv.putText(
                vis,
                f"id={tid}",
                (int(x) + 5, int(y) - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv.LINE_AA,
            )

            pts = np.asarray(
                [(hx, hy) for _, hx, hy, _ in tr.history[-50:]],
                dtype=np.int32,
            )

            if len(pts) >= 2:
                cv.polylines(
                    vis,
                    [pts.reshape(-1, 1, 2)],
                    False,
                    color,
                    1,
                )

        if self.roi is not None:
            x, y, w, h = self.roi
            cv.rectangle(
                vis,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                1,
            )

        return vis

    @staticmethod
    def _color_by_id(track_id: int):
        rng = np.random.default_rng(track_id)
        color = rng.integers(80, 255, size=3)
        return int(color[0]), int(color[1]), int(color[2])

    def tracks_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_tracks(self, path: str):
        df = self.tracks_dataframe()
        df.to_csv(path, index=False)
        return df
    
