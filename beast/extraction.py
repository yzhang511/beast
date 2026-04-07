from __future__ import annotations

from pathlib import Path
from typing import Literal

from tqdm import tqdm

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typeguard import typechecked

from beast.video import (
    compute_video_motion_energy,
    get_frames_from_idxs,
)


@typechecked
def extract_frames(
    input_path: Path | str,
    output_dir: Path | str,
    frames_per_video: int = 500,
    method: str = 'pca_kmeans',
    num_workers: int = 8,
    timestamp_dir: Path | str | None = None,
    neural_data_dir: Path | str | None = None,
) -> dict:
    """Extract representative frames from videos using intelligent sampling methods.

    Processes all video files in a directory and extracts a subset of frames using
    advanced selection algorithms to capture diverse visual content. Each video's
    frames are saved to a separate subdirectory named after the video file.

    Parameters
    ----------
    input_path: directory containing video files (.mp4 and .avi formats supported)
    output_dir: directory where extracted frames will be saved; creates subdirectories
        for each video named after the video filename (without extension)
    frames_per_video: maximum number of frames to extract from each video
    method: frame selection method; currently supported:
      - 'pca_kmeans': Uses PCA dimensionality reduction followed by k-means clustering to select
        diverse, representative frames
      - 'precomputed': Loads precomputed frame indices from a file in the output directory
        named 'selected_frame_indices.txt'
      - 'timestamp': Select frame indices from provided time intervals; requires `timestamp_dir` and `neural_data_dir`
    num_workers: number of parallel workers for processing (currently unused but reserved
      for future parallel processing implementation)
    timestamp_dir: directory containing video timestamps from each session
    neural_data_dir: directory containing neural and behavior data from each session

    Returns
    -------
    Summary statistics containing:
        - 'total_frames': Total number of frames extracted across all videos
        - 'total_videos': Number of videos processed

    Examples
    --------
    Extract frames from all videos in a directory:

    >>> results = extract_frames(
    ...     input_path="./videos",
    ...     output_dir="./extracted_frames",
    ...     frames_per_video=300,
    ...     method='pca_kmeans'
    ... )
    >>> print(f"Extracted {results['total_frames']} frames from {results['total_videos']} videos")

    Process fewer frames per video:

    >>> results = extract_frames(
    ...     input_path=Path("./raw_videos"),
    ...     output_dir=Path("./training_data"),
    ...     frames_per_video=100
    ... )

    Directory Structure
    -------------------
    Input directory:
        input_path/
        ├── video1.mp4
        ├── video2.avi
        └── video3.mp4

    Output directory:
        output_dir/
        ├── video1/
        │   ├── selected_frame_indices.txt
        │   ├── frame_001.png
        │   ├── frame_045.png
        │   └── ...
        ├── video2/
        │   ├── selected_frame_indices.txt
        │   ├── frame_012.png
        │   └── ...
        └── video3/
            └── ...

    Timestamp directory:
        timestamp_dir/
        ├── _ibl_leftCamera.times.eid1.npy
        ├── _ibl_leftCamera.times.eid2.npy
        └── ...

    Neural data directory:
        neural_data_dir/
        ├── eid1_aligned.npz
        ├── eid2_aligned.npz
        └── ...

    Notes
    -----
    - Only processes .mp4 and .avi video files
    - The PCA-kmeans method resizes frames to 32x32 pixels for analysis, but exports frames at
      original resolution
    - Context frames (neighboring frames) are included in the selection
    - Progress information is logged during processing
    - Frame selection aims to maximize visual diversity while avoiding redundant or similar frames

    """

    print(f'Extracting frames from: {input_path}')
    print(f'Saving to: {output_dir}')
    print(f'Method: {method}')
    print(f'Frames per video: {frames_per_video}')

    video_files = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi'))
    total_videos = 0
    total_frames = 0
    for video_file in video_files:

        n_digits = 8
        extension = 'png'
        save_dir = output_dir.joinpath(video_file.stem)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if method == 'pca_kmeans':
            idxs = select_frame_idxs_kmeans(
                video_file=video_file,
                resize_dims=32,
                n_frames_to_select=frames_per_video,
            )
        elif method == 'precomputed':
            # TODO: hard-coded for now; make this more general
            selected_frames_dir = Path(str(save_dir).replace('rightCamera', 'leftCamera'))
            idxs = np.loadtxt(
                selected_frames_dir / 'selected_anchor_frames.txt', dtype=int
            ).flatten()
            print(f'Loaded precomputed frame indices from: {selected_frames_dir / "selected_anchor_frames.txt"}')
            print(f'Selected {len(idxs)} frames')
        elif method == 'timestamp':
            idxs_dict = select_frame_idxs_timestamp(
                video_file=video_file,
                timestamp_dir=timestamp_dir,
                neural_data_dir=neural_data_dir,
            )
        else:
            raise NotImplementedError

        if method in ['pca_kmeans', 'precomputed']:
            # save anchor + context frame indices
            idx_path = save_dir / 'selected_anchor_frames.txt'
            np.savetxt(idx_path, idxs, fmt='%d')
            print(f'Saved selected anchor frame indices to: {idx_path}')
            total_frame_idxs = export_frames(
                video_file=video_file,
                output_dir=save_dir,
                frame_idxs=idxs,
                context_frames=1,
                n_digits=n_digits,
                extension=extension,
            )
            # save csv file inside same output directory
            frames_to_label = np.array([
                "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in total_frame_idxs
            ])
            csv_path = save_dir / 'selected_total_frames.csv'
            np.savetxt(csv_path, np.sort(frames_to_label), delimiter=',', fmt='%s')
            print(f'Saved selected frame list to: {csv_path}')
            total_frames += len(idxs)
        elif method == 'timestamp':
            for split, idxs in idxs_dict.items():
                frames_to_label = []
                for interval_idx, idx in tqdm(enumerate(idxs), total=len(idxs), desc=f'Exporting frames for {split}'):
                    _ = export_frames(
                        video_file=video_file,
                        output_dir=save_dir.joinpath(split),
                        frame_idxs=idx,
                        context_frames=0,
                        n_digits=n_digits,
                        extension=extension,
                        interval_idx=interval_idx,
                    )
                    total_frames += len(idx)
                    frames_to_label.extend(
                        "img%s.%s" % (str(_idx).zfill(n_digits), extension) for _idx in idx
                    )
                # save csv file inside same output directory
                frames_to_label = np.array(frames_to_label)
                csv_path = save_dir / split / 'selected_total_frames.csv'
                np.savetxt(csv_path, np.sort(frames_to_label), delimiter=',', fmt='%s')
                print(f'Saved selected frame list to: {csv_path}')
        else:
            raise NotImplementedError

        total_videos += 1

    return {
        'total_frames': total_frames,
        'total_videos': total_videos,
    }


def interval_to_frame_indices(
    intervals: np.ndarray,
    timestamps: np.ndarray,
    *,
    video_fps: float = 60.0,
) -> np.ndarray:
    """For each row ``[t_lo, t_hi]``, return frame indices with ``t_lo <= t < t_hi`` (seconds).

    ``timestamps`` are per-frame times in seconds (e.g. IBL ``_ibl_*Camera.times``).
    Output is ``dtype=object``; entry ``i`` is a 1D ``int64`` array of frame indices.

    Indices are truncated to at most ``round((t_hi - t_lo) * video_fps)`` per interval so
    boundary or FP noise cannot yield extra frames (e.g. 61 timestamps in a 1 s bin → 60).
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f'Expected intervals of shape (n, 2), got {intervals.shape}')
    timestamps = np.asarray(timestamps, dtype=np.float64).ravel()
    rows: list[np.ndarray] = []
    for t_lo, t_hi in intervals:
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        idxs = np.flatnonzero(mask).astype(np.int64, copy=False)
        max_count = int(round((float(t_hi) - float(t_lo)) * video_fps))
        if max_count > 0 and idxs.size > max_count:
            idxs = idxs[:max_count]
        rows.append(idxs)
    return np.asarray(rows, dtype=object)


@typechecked
def select_frame_idxs_timestamp(
    video_file: str | Path,
    timestamp_dir: Path | str,
    neural_data_dir: Path | str,
) -> dict:
    """Map train / val / test time intervals (seconds) to video frame indices per interval.

    Loads ``train_intervals``, ``val_intervals``, ``test_intervals`` from
    ``{eid}_aligned.npz`` and camera frame times from
    ``_ibl_{left|right}Camera.times.{eid}.npy``.

    Returns
    -------
    train_idxs, val_idxs, test_idxs
        Each is ``np.ndarray`` with ``dtype=object``; element ``i`` is a 1D int array of
        frame indices falling in ``intervals[i]`` (half-open ``[t_lo, t_hi)`` in seconds).
    """
    timestamp_dir = Path(timestamp_dir)
    neural_data_dir = Path(neural_data_dir)
    eid = video_file.stem.split('.')[-1]
    npz_path = neural_data_dir / eid / f'{eid}_aligned.npz'
    ts_path = timestamp_dir / f'_ibl_leftCamera.times.{eid}.npy'
    if not ts_path.is_file():
        raise FileNotFoundError(f'Missing camera timestamps: {ts_path}')

    with np.load(npz_path) as data:
        train_intervals = data['train_intervals']
        val_intervals = data['val_intervals']
        test_intervals = data['test_intervals']

    timestamps = np.load(ts_path)

    train_idxs = interval_to_frame_indices(train_intervals, timestamps)
    val_idxs = interval_to_frame_indices(val_intervals, timestamps)
    test_idxs = interval_to_frame_indices(test_intervals, timestamps)
    return {
        'train': train_idxs,
        'val': val_idxs,
        'test': test_idxs,
    }


@typechecked
def _run_kmeans(data: np.ndarray, n_clusters: int, seed: int = 0) -> tuple:
    np.random.seed(seed)
    kmeans_obj = KMeans(n_clusters, n_init='auto')
    kmeans_obj.fit(data)
    cluster_labels = kmeans_obj.labels_
    cluster_centers = kmeans_obj.cluster_centers_
    return cluster_labels, cluster_centers


@typechecked
def select_frame_idxs_kmeans(
    video_file: str | Path,
    resize_dims: int = 64,
    n_frames_to_select: int = 20,
    frame_range: list = [0, 1],
) -> np.ndarray:
    """Select distinct frames during movement using kmeans on motion-energy thresholded frame PCs.

    Parameters
    ----------
    video_file: absolute path to video file
    resize_dims: number of pixels (in both dimensions) to downsample video before computing motion
        energy; exported frames will retain original resolution
    n_frames_to_select: number of anchor frames to select per video
    frame_range: define range of video considered for frame extraction; for example, [0, 1] uses
        the full video, while [0.25, 0.75] uses the central 50% of the video

    Returns
    -------
    frames array of shape (n_frames_to_select, n_channels, ypix, xpix)

    """

    # check inputs
    assert frame_range[0] >= 0
    assert frame_range[1] <= 1

    # read all frames, reshape, chop off unwanted portions of beginning/end
    print('computing motion energy...')
    me, frames = compute_video_motion_energy(
        video_file=video_file,
        resize_dims=resize_dims,
        return_frames=True,
    )
    frame_count = me.shape[0]
    beg_frame = int(float(frame_range[0]) * frame_count)
    end_frame = int(float(frame_range[1]) * frame_count) - 2  # leave room for context
    assert (end_frame - beg_frame) >= n_frames_to_select, 'valid video segment too short!'

    # find high me frames, defined as those with me larger than nth percentile me
    prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
    idxs_high_me = np.where(me > np.percentile(me, prctile))[0]
    # just use all frames if the user wants to extract a large fraction of the frames
    # (helpful for very short videos)
    if len(idxs_high_me) < n_frames_to_select:
        idxs_high_me = np.arange(me.shape[0])

    # compute pca over high me frames
    print('performing pca over high motion energy frames...')
    pca_obj = PCA(n_components=np.min([frames[idxs_high_me].shape[0], 32]))
    embedding = pca_obj.fit_transform(X=frames[idxs_high_me])
    del frames  # free up memory

    # cluster low-d pca embeddings
    print('performing kmeans clustering...')
    _, centers = _run_kmeans(data=embedding, n_clusters=n_frames_to_select)
    # centers is initially of shape (n_clusters, n_pcs); reformat
    centers = centers.T[None, :]

    # find high me frame that is closest to each cluster center
    # embedding is shape (n_frames, n_pcs)
    # centers is shape (1, n_pcs, n_clusters)
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    # dists is shape (n_frames, n_clusters)
    idxs_prototypes_ = np.argmin(dists, axis=0)
    # now index into high me frames to get overall indices, add offset
    idxs_prototypes = idxs_high_me[idxs_prototypes_] + beg_frame

    return idxs_prototypes


@typechecked
def export_frames(
    video_file: str | Path,
    output_dir: str | Path,
    frame_idxs: np.ndarray,
    extension: str = 'png',
    n_digits: int = 8,
    context_frames: int = 1,
    interval_idx: int | None = None,
) -> np.ndarray:
    """Export selected frames from a video to individual png files.

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    output_dir: absolute path to parent directory in which selected frames are saved
    frame_idxs: indices of frames to export
    extension: only 'png' currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save
    interval_idx: index of interval within split
    """

    # expand frame_idxs to include context frames
    if context_frames > 0:
        cap = cv2.VideoCapture(str(video_file))
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)
        cap.release()

    # load frames from video
    frames = get_frames_from_idxs(video_file, frame_idxs)

    # save out frames
    output_dir.mkdir(parents=True, exist_ok=True)
    for tbin_idx, (frame, idx) in enumerate(zip(frames, frame_idxs)):
        if interval_idx is not None:
            filename = str(output_dir.joinpath(f'interval{interval_idx}timebin{tbin_idx}.{extension}'))
        else:
            filename = str(output_dir.joinpath(f'img{str(idx).zfill(n_digits)}.{extension}'))
        
        cv2.imwrite(
            filename=filename,
            img=cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR),
        )

    return frame_idxs
