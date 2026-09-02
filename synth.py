import numpy as np
import cv2 as cv
from StatTools.generators.ndfnoise_generator import ndfnoise
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from IPython.display import clear_output
from collections import defaultdict
import seaborn as sns

def draw_traj(trajectories, frame_shape: tuple, ants_num: int):
    
    colors = sns.color_palette(palette='bright', n_colors=ants_num)
    colors = [tuple([int(c*255) for c in color]) for color in colors]

    bg = np.zeros(shape=(*frame_shape, 3), dtype=np.uint8)
    trajs = []
    for ant in range(trajectories.shape[1]):
        trajs.append(np.array([np.array(traj) for traj in trajectories[:, ant]]))
    for ant, color in  zip(trajs, colors):

        cv.polylines(bg, [ant], isClosed=False, color=color, thickness=2)

    plt.imshow(cv.cvtColor(bg, cv.COLOR_BGR2RGB))
    plt.show()


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
        start_x = np.random.randint(-margin, frame_shape[1] + margin, (ants_num,))
        start_y = np.random.randint(-margin, frame_shape[0] + margin, (ants_num,))
        start_point = np.stack([start_x, start_y], axis=1)

    trajectories = np.stack([x, y], axis=2)
    trajectories = trajectories + start_point

    return trajectories


def draw_ants_with_direction(trajectories, output_path:str, frame_num: int, ants_num: int, frame_shape: tuple, ant_length = 5, ant_width=2, smooth_window:int=3, imshow:bool=False):
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_shape[1], frame_shape[0])
    if output_path:
        out = cv.VideoWriter(output_path, fourcc, 24.0, frame_size, isColor=True)
    for frame_idx in range(frame_num):
        # frame = gen_poisson_bg(frame_shape, sense=0.43)
        frame = np.ones((*frame_shape, 3), dtype=np.uint8)*255
        
        for ant_idx in range(ants_num):
            
            x, y = trajectories[frame_idx, ant_idx]
            start_index = max(0, frame_idx-smooth_window)
            dx = x - trajectories[start_index:frame_idx, ant_idx, 0].mean()
            dy = y - trajectories[start_index:frame_idx, ant_idx, 1].mean()
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            center = (int(x), int(y))
            axes = (ant_length, ant_width)
            cv.ellipse(frame, center, axes, angle, 0, 360, 
                       (100, 100, 100), -1)
        frame = cv.GaussianBlur(frame, ksize=(7,7), sigmaX=1.0)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if output_path:
            out.write(frame)

        if imshow:
            # cv2.imshow('frame', frame)
            plt.figure(figsize=(12, 8))
            plt.imshow(frame)
            plt.axis('off')
            plt.show()
            
            clear_output(wait=True)
            if cv.waitKey(1) & 0xFF == 27:
                cv.destroyAllWindows()
                if output_path:
                    out.release()
                break

    if output_path:
        out.release()
    cv.destroyAllWindows()

def _as_radii(radii: Union[float, np.ndarray], n: int) -> np.ndarray:
    """
    Приводит радиусы к массиву длины n.
    """
    if np.isscalar(radii):
        return np.full(n, float(radii), dtype=np.float32)

    radii = np.asarray(radii, dtype=np.float32)

    if radii.size == 1:
        return np.full(n, float(radii.ravel()[0]), dtype=np.float32)

    if radii.shape != (n,):
        raise ValueError("radii должен быть скаляром или массивом длины ants_num")

    return radii


def _clip_to_bounds(
    positions: np.ndarray,
    bounds: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Обрезает позиции по заданным границам.

    bounds = (xmin, xmax, ymin, ymax)
    """
    if bounds is None:
        return positions

    xmin, xmax, ymin, ymax = bounds

    positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
    positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)

    return positions


def _neighbor_pairs(positions: np.ndarray, cell_size: float):
    """
    Возвращает потенциально близкие пары особей с использованием
    пространственной сетки.

    Это нужно, чтобы не перебирать все пары O(N^2).
    """
    grid = defaultdict(list)

    for i, p in enumerate(positions):
        key = (
            int(np.floor(p[0] / cell_size)),
            int(np.floor(p[1] / cell_size)),
        )
        grid[key].append(i)

    # Смещения соседних ячеек, чтобы каждую пару проверить один раз.
    neighbor_offsets = (
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    )

    for (cx, cy), indices in grid.items():
        m = len(indices)

        # Пары внутри одной ячейки.
        for a in range(m):
            i = indices[a]
            for b in range(a + 1, m):
                j = indices[b]
                yield i, j

        # Пары с соседними ячейками.
        for dx, dy in neighbor_offsets:
            other = grid.get((cx + dx, cy + dy))
            if other is None:
                continue

            for i in indices:
                for j in other:
                    yield i, j


def _separate_positions(
    positions: np.ndarray,
    radii: np.ndarray,
    max_iter: int = 8,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    rng: Optional[np.random.Generator] = None,
    relaxation: float = 1.0,
) -> np.ndarray:
    """
    Итеративно разводит позиции особей так, чтобы расстояние между центрами
    было не меньше r_i + r_j.
    """
    positions = np.asarray(positions, dtype=np.float32)
    pos = positions.copy()

    pos = _clip_to_bounds(pos, bounds)

    n = pos.shape[0]

    if n < 2:
        return pos

    if max_iter <= 0:
        return pos

    max_radius = float(np.max(radii))

    if max_radius <= 0.0:
        return pos

    if rng is None:
        rng = np.random.default_rng()

    # Размер ячейки выбираем по максимальному диаметру.
    cell_size = max(1.0, 2.0 * max_radius)

    for _ in range(max_iter):
        moved = False

        for i, j in _neighbor_pairs(pos, cell_size):
            min_dist = float(radii[i] + radii[j])

            if min_dist <= 0.0:
                continue

            delta = pos[j] - pos[i]
            dist2 = float(delta[0] * delta[0] + delta[1] * delta[1])
            min_dist2 = min_dist * min_dist

            if dist2 < min_dist2:
                moved = True

                dist = float(np.sqrt(dist2))

                if dist < 1e-6:
                    # Если точки совпадают, задаем случайное направление.
                    angle = float(rng.random()) * 2.0 * np.pi
                    direction = np.array(
                        [np.cos(angle), np.sin(angle)],
                        dtype=np.float32,
                    )
                else:
                    direction = delta / dist

                overlap = min_dist - dist

                # Каждая особь смещается на половину перекрытия.
                correction = 0.5 * relaxation * overlap * direction

                pos[i] -= correction
                pos[j] += correction

        pos = _clip_to_bounds(pos, bounds)

        if not moved:
            break

    return pos


def make_frame_bounds(
    frame_shape: tuple,
    margin: float = 0.0,
    inside: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Возвращает границы области в формате:

        (xmin, xmax, ymin, ymax)

    Параметры:
    -----------
    frame_shape : tuple
        (height, width)

    margin : float
        Отступ от границ кадра.

    inside : bool
        Если True, ограничивает траектории внутри кадра с отступом margin:
            [margin, width - margin - 1]

        Если False, разрешает находиться в расширенной области:
            [-margin, width + margin - 1]

        Это полезно, если объекты могут заходить в кадр и выходить из него.
    """
    height, width = frame_shape

    if inside:
        xmin = margin
        xmax = width - margin - 1
        ymin = margin
        ymax = height - margin - 1
    else:
        xmin = -margin
        xmax = width + margin - 1
        ymin = -margin
        ymax = height + margin - 1

    if xmin > xmax or ymin > ymax:
        raise ValueError("Некорректные границы после вычисления")

    return float(xmin), float(xmax), float(ymin), float(ymax)


def postprocess_collisions(
    trajectories: np.ndarray,
    radii: Union[float, np.ndarray] = 1.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    max_iter: int = 8,
    substeps: int = 1,
    relaxation: float = 1.0,
    seed: Optional[int] = None,
    return_float: bool = False,
) -> np.ndarray:
    """
    Постобработка траекторий для устранения столкновений особей.

    Параметры:
    -----------
    trajectories : np.ndarray
        Массив формы (frame_num, ants_num, 2).

    radii : float или np.ndarray
        Радиус тела особи. Если нужно, чтобы центры двух особей
        не подходили ближе чем D пикселей, задайте радиус = D / 2.

        Для разных радиусов у разных особей передайте массив длины ants_num.

    bounds : tuple или None
        Границы области: (xmin, xmax, ymin, ymax).
        Если None, границы не учитываются.

    max_iter : int
        Число итераций разрешения пересечений на каждом шаге.

    substeps : int
        Число подшагов между соседними кадрами.

        Это полезно, если скорости большие и особи могут "проскакивать"
        друг сквозь друга между кадрами.

        Например, substeps=2 или substeps=4 уменьшают туннелирование.

    relaxation : float
        Коэффициент релаксации. Обычно 1.0.
        Можно поставить 0.7-0.9 для более мягкой коррекции.

    seed : int или None
        Seed для воспроизводимости.

    return_float : bool
        Если True, возвращает float-траектории.
        Если False, возвращает int после округления.
    """
    raw = np.asarray(trajectories, dtype=np.float32).copy()

    if raw.ndim != 3 or raw.shape[2] != 2:
        raise ValueError("trajectories должен иметь форму (frame_num, ants_num, 2)")

    frame_num, ants_num, _ = raw.shape

    if frame_num == 0 or ants_num == 0:
        if return_float:
            return raw
        return np.round(raw).astype(np.int32)

    radii = _as_radii(radii, ants_num)
    rng = np.random.default_rng(seed)

    substeps = max(1, int(substeps))

    out = np.empty_like(raw)

    # Начальный кадр.
    out[0] = _separate_positions(
        raw[0],
        radii=radii,
        max_iter=max_iter,
        bounds=bounds,
        rng=rng,
        relaxation=relaxation,
    )

    # Последующие кадры.
    for t in range(1, frame_num):
        # Исходное приращение из сгенерированной траектории.
        desired_delta = raw[t] - raw[t - 1]

        # Двигаем уже скорректированную предыдущую позицию.
        current_pos = out[t - 1].copy()

        step_delta = desired_delta / float(substeps)

        for _ in range(substeps):
            proposed_pos = current_pos + step_delta

            proposed_pos = _separate_positions(
                proposed_pos,
                radii=radii,
                max_iter=max_iter,
                bounds=bounds,
                rng=rng,
                relaxation=relaxation,
            )

            current_pos = proposed_pos

        out[t] = current_pos

    if return_float:
        return out

    return np.round(out).astype(np.int32)


from typing import Tuple

import numpy as np
import pandas as pd


def build_visible_gt(
    trajectories: np.ndarray,
    frame_shape: Tuple[int, int],
    border_margin: float = 0.0,
    min_visible_frames: int = 1,
    first_id: int = 1,
) -> pd.DataFrame:
    """
    Строит ground truth для видимых сегментов траекторий.

    Параметры
    ---------
    trajectories : np.ndarray
        Массив формы (frames, ants, 2), последняя координата (x, y).
    frame_shape : Tuple[int, int]
        Форма кадра в виде (height, width).
    border_margin : float
        Внутренний отступ от границ кадра.
        Если 0, объект считается видимым, когда его центр внутри кадра.
    min_visible_frames : int
        Минимальная длина видимого сегмента, чтобы попасть в GT.
        Полезно, если нужно отбросить однокадровые появления.
    first_id : int
        С какого ID нумеровать треки. Обычно 1.

    Возвращает
    ----------
    pd.DataFrame со столбцами:
        frame, track_id, physical_ant_id, x, y
    """

    if trajectories.size == 0:
        return pd.DataFrame(
            columns=["frame", "track_id", "physical_ant_id", "x", "y"]
        )

    T, N, _ = trajectories.shape

    # frame_shape задан как (height, width)
    H, W = frame_shape

    x = trajectories[..., 0]
    y = trajectories[..., 1]

    # Объект считается видимым, если его координаты конечны
    # и центр находится внутри кадра.
    # Если нужно учитывать частичную видимость тела, см. комментарии ниже.
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= border_margin)
        & (x < W - border_margin)
        & (y >= border_margin)
        & (y < H - border_margin)
    )

    # Сдвиг маски видимости на один кадр назад.
    # Для первого кадра предыдущего состояния нет.
    prev_valid = np.roll(valid, 1, axis=0)
    if T > 0:
        prev_valid[0, :] = False

    # Начало нового видимого сегмента:
    # текущий кадр видим, предыдущий был невидим.
    segment_start = valid & ~prev_valid

    # Локальный номер сегмента внутри каждого физического муравья.
    # Например:
    # valid:         [False, True, True, False, True]
    # segment_start: [False, True, False, False, True]
    # local_segment: [0, 1, 1, 0, 2]
    local_segment = np.cumsum(segment_start, axis=0)
    local_segment = np.where(valid, local_segment, 0)

    # Количество сегментов у каждого физического муравья.
    if T > 0:
        num_segments = local_segment.max(axis=0)
    else:
        num_segments = np.zeros(N, dtype=np.int64)

    # Временные глобальные ID сегментов.
    # Они нужны только для последующего переименования в порядке появления.
    offsets = np.zeros(N, dtype=np.int64)
    if N > 1:
        offsets[1:] = np.cumsum(num_segments[:-1])

    temp_track_id = np.where(
        valid,
        offsets[None, :] + local_segment,
        -1,
    )

    # Собираем только видимые строки.
    frames, ants = np.where(valid)

    df = pd.DataFrame(
        {
            "frame": frames,
            "temp_track_id": temp_track_id[frames, ants],
            "physical_ant_id": ants,
            "x": trajectories[frames, ants, 0],
            "y": trajectories[frames, ants, 1],
        }
    )

    if df.empty:
        return pd.DataFrame(
            columns=["frame", "track_id", "physical_ant_id", "x", "y"]
        )

    # Опционально: удалить слишком короткие сегменты.
    if min_visible_frames > 1:
        counts = df.groupby("temp_track_id")["frame"].transform("size")
        df = df[counts >= min_visible_frames].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["frame", "track_id", "physical_ant_id", "x", "y"]
        )

    # Переименовываем временные ID в порядке первого появления.
    # Это делает track_id последовательным и читаемым.
    df = df.sort_values(["frame", "temp_track_id"], kind="mergesort")
    df["track_id"] = pd.factorize(df["temp_track_id"])[0] + first_id
    df = df.drop(columns="temp_track_id")

    df = df[
        [
            "frame",
            "track_id",
            "physical_ant_id",
            "x",
            "y",
        ]
    ]

    df = df.sort_values(["frame", "track_id"], kind="mergesort").reset_index(drop=True)

    return df

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