# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, SegmentPoseMetrics, kpt_iou, mask_iou


class SegmentPoseValidator(DetectionValidator):
    """Validator for joint segmentation and pose estimation models.

    Evaluates detection, segmentation, and pose estimation simultaneously,
    computing metrics for all three tasks: box mAP, mask mAP, and keypoint mAP.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize SegmentPoseValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "segment_pose"
        self.metrics = SegmentPoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting masks and keypoints to float."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics for segmentation and pose evaluation."""
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process predictions to extract boxes, masks, and keypoints."""
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]
        for i, pred in enumerate(preds):
            extra = pred.pop("extra")
            # Split extra into mask coefficients and keypoints
            nm = proto.shape[1]  # number of mask channels
            nk = self.kpt_shape[0] * self.kpt_shape[1]
            mask_coeff = extra[:, :nm]
            kpt_data = extra[:, nm:]

            # Process masks
            pred["masks"] = (
                self.process(proto[i], mask_coeff, pred["bboxes"], shape=imgsz)
                if mask_coeff.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
            # Process keypoints
            pred["keypoints"] = kpt_data.view(-1, *self.kpt_shape) if kpt_data.shape[0] else kpt_data
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch with masks and keypoints for validation."""
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]

        # Masks
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks

        # Keypoints
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = prepared_batch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        prepared_batch["keypoints"] = kpts

        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Compute correct prediction matrix for boxes, masks, and keypoints."""
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]

        # Mask metrics
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})

        # Pose metrics
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})

        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot batch predictions with masks and bounding boxes."""
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        scaled = super().scale_preds(predn, pbatch)
        scaled["masks"] = ops.scale_masks(
            predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )[0].byte()
        scaled["kpts"] = ops.scale_coords(
            pbatch["imgsz"], predn["keypoints"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return scaled
