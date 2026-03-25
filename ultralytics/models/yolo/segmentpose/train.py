# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentPoseModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


class SegmentPoseTrainer(yolo.detect.DetectionTrainer):
    """Trainer for joint segmentation and pose estimation models.

    This trainer extends DetectionTrainer to handle simultaneous instance segmentation and keypoint estimation,
    combining both tasks into a unified training pipeline.

    Attributes:
        loss_names (tuple[str]): Names of the 6 loss components used during training.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize SegmentPoseTrainer with task set to 'segment_pose'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment_pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> SegmentPoseModel:
        """Return SegmentPoseModel initialized with specified config and weights."""
        model = SegmentPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute on the model."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of SegmentPoseValidator for validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss"
        return yolo.segmentpose.SegmentPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. Required for segment_pose task.")
        return data
