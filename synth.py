import numpy as np
import cv2 as cv
from StatTools.generators.ndfnoise_generator import ndfnoise

def draw_traj(trajectories,
              frame_shape: tuple,
              thickness: int):
    
    ants_num = trajectories.shape[1]
    bgs = [np.zeros(shape=frame_shape, dtype=np.uint8) for _ in range(ants_num)]
    
    for ant_idx in range(ants_num):
        
        traj_raw = trajectories[:, ant_idx, :]
        
        valid_mask = ~np.any(np.isnan(traj_raw), axis=1)
        traj_clean = traj_raw[valid_mask]
        
        if len(traj_clean) < 2:
            continue
         
        traj_int = np.ascontiguousarray(traj_clean.astype(np.int32))
        
        cv.polylines(bgs[ant_idx], [traj_int], isClosed=False, color=255, thickness=thickness)
    
    return bgs


def gen_traj(frame_num: int,
             ants_num: int,
             frame_shape: tuple,
             margin:int = 10,
             hurst_move: float = 0.5,
             hurst_species: float = 0.5, 
             start_point = None):
    

    dx = ndfnoise(shape=(frame_num, ants_num), hurst=[hurst_move, hurst_species], normalize=True, dtype=np.float32)
    dy = ndfnoise(shape=(frame_num, ants_num), hurst=[hurst_move, hurst_species], normalize=True, dtype=np.float32)

    x = dx.round().cumsum(axis=0).astype(np.int32)
    y = dy.round().cumsum(axis=0).astype(np.int32)

    if start_point is None:
        start_x = np.random.randint(margin, frame_shape[1] - margin, (ants_num,))
        start_y = np.random.randint(margin, frame_shape[0] - margin, (ants_num,))
        start_point = np.stack([start_x, start_y], axis=1)

    trajectories = np.stack([x, y], axis=2)
    trajectories = trajectories + start_point

    return trajectories


def calc_iou(tracks, thickness, frame_shape):

    union = np.zeros(frame_shape, dtype=np.uint16)
    canvas = np.zeros(frame_shape, dtype=np.uint8)

    ants_num = tracks.shape[1]

    for ant_idx in range(ants_num):

        traj = tracks[:, ant_idx]

        valid = ~np.any(np.isnan(traj), axis=1)
        traj = traj[valid]

        if len(traj) < 2:
            continue

        canvas.fill(0)

        cv.polylines(
            canvas,
            [np.ascontiguousarray(traj, dtype=np.int32)],
            isClosed=False,
            color=1,
            thickness=thickness,
        )

        union += canvas

    
    union_mask = union > 0
    intersection = union - union_mask
    iou = intersection.sum() / union.sum()

    return iou, intersection, union