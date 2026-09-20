import numpy as np
import pandas as pd
import motmetrics as mm


def evaluate_mot(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    distance_threshold: float = 25.0,
):
    """
    Evaluate multi-object tracking using py-motmetrics.
+
    Parameters
    ----------
    gt : pd.DataFrame
        Ground-truth trajectories.

        Required columns:
            frame
            track_id
            x
            y

    pred : pd.DataFrame
        Tracker output.

        Required columns:
            frame
            track_id
            x
            y

    distance_threshold : float
        Maximum Euclidean distance in pixels for a GT/prediction
        correspondence.

        A GT-prediction pair with distance larger than this value
        is considered invalid for matching.

    Returns
    -------
    summary : pd.DataFrame
        MOT metrics.

    accumulator : mm.MOTAccumulator
        Full motmetrics accumulator containing frame-level events.
    """

    required_columns = {"frame", "track_id", "x", "y"}

    missing_gt = required_columns - set(gt.columns)
    missing_pred = required_columns - set(pred.columns)

    if missing_gt:
        raise ValueError(
            f"GT is missing columns: {sorted(missing_gt)}"
        )

    if missing_pred:
        raise ValueError(
            f"Prediction is missing columns: {sorted(missing_pred)}"
        )

    gt = gt.copy()
    pred = pred.copy()

    # Remove invalid rows
    gt = gt.dropna(subset=["frame", "track_id", "x", "y"])
    pred = pred.dropna(subset=["frame", "track_id", "x", "y"])

    # Ensure correct dtypes
    gt["frame"] = gt["frame"].astype(int)
    pred["frame"] = pred["frame"].astype(int)

    gt["track_id"] = gt["track_id"].astype(int)
    pred["track_id"] = pred["track_id"].astype(int)

    # MOTAccumulator stores all frame-level associations.
    accumulator = mm.MOTAccumulator(auto_id=True)

    # We must evaluate every frame appearing either in GT or prediction.
    frames = sorted(
        set(gt["frame"].unique()) |
        set(pred["frame"].unique())
    )

    for frame_id in frames:

        gt_frame = gt[gt["frame"] == frame_id]
        pred_frame = pred[pred["frame"] == frame_id]

        gt_ids = gt_frame["track_id"].to_numpy()
        pred_ids = pred_frame["track_id"].to_numpy()

        gt_points = gt_frame[["x", "y"]].to_numpy(dtype=np.float32)
        pred_points = pred_frame[["x", "y"]].to_numpy(dtype=np.float32)

        # No GT and no predictions.
        if len(gt_points) == 0 and len(pred_points) == 0:
            accumulator.update(
                [],
                [],
                []
            )
            continue

        # GT exists, predictions don't.
        if len(pred_points) == 0:
            accumulator.update(
                gt_ids,
                [],
                np.empty((len(gt_ids), 0))
            )
            continue

        # Predictions exist, GT doesn't.
        if len(gt_points) == 0:
            accumulator.update(
                [],
                pred_ids,
                np.empty((0, len(pred_ids)))
            )
            continue

        # Pairwise Euclidean distance matrix.
        distances = np.linalg.norm(
            gt_points[:, None, :] -
            pred_points[None, :, :],
            axis=2,
        )

        # Distances larger than the threshold are forbidden matches.
        distances[distances > distance_threshold] = np.nan

        accumulator.update(
            gt_ids,
            pred_ids,
            distances,
        )

    # Metrics requested.
    metrics = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_misses",
        "num_false_positives",
        "num_switches",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_fragmentations",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        'deta_alpha',
        'assa_alpha',
        'hota_alpha'
    ]

    mh = mm.metrics.create()

    summary = mh.compute(
        accumulator,
        metrics=metrics,
        name="tracker",
    )

    return summary, accumulator