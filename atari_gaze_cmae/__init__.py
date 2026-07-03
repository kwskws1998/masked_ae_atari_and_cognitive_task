"""Atari-HEAD data and frame-conditioned gaze MAE models."""

from .active_gaze_dt import (
    ActiveGazeBehaviorCloner,
    ActiveGazeBehaviorClonerOutput,
    ActiveGazeDecisionTransformer,
    ActiveGazeDecisionTransformerConfig,
    ActiveGazeDecisionTransformerOutput,
    ActiveGazeMAEVisualEncoder,
    ActiveGazeVisualEncoderOutput,
    GazeMaskedDecisionTransformer,
)
from .data import (
    AtariHeadLabel,
    AtariHeadTrialDataset,
    collate_atari_head_samples,
    gaze_points_to_heatmap,
    read_atari_head_labels,
)
from .model import AtariHeadGazeMAE, AtariHeadGazeMAEConfig, AtariHeadGazeMAEOutput
from .trajectory_data import AtariHeadHDF5TrajectoryDataset, select_hdf5_groups
from .zenodo import (
    ZENODO_RECORD_ID,
    ZenodoFile,
    download_files,
    fetch_zenodo_manifest,
    files_for_trial,
)

__all__ = [
    "ActiveGazeBehaviorCloner",
    "ActiveGazeBehaviorClonerOutput",
    "ActiveGazeDecisionTransformer",
    "ActiveGazeDecisionTransformerConfig",
    "ActiveGazeDecisionTransformerOutput",
    "ActiveGazeMAEVisualEncoder",
    "ActiveGazeVisualEncoderOutput",
    "AtariHeadGazeMAE",
    "AtariHeadGazeMAEConfig",
    "AtariHeadGazeMAEOutput",
    "AtariHeadHDF5TrajectoryDataset",
    "AtariHeadLabel",
    "AtariHeadTrialDataset",
    "GazeMaskedDecisionTransformer",
    "ZENODO_RECORD_ID",
    "ZenodoFile",
    "collate_atari_head_samples",
    "download_files",
    "fetch_zenodo_manifest",
    "files_for_trial",
    "gaze_points_to_heatmap",
    "read_atari_head_labels",
    "select_hdf5_groups",
]
