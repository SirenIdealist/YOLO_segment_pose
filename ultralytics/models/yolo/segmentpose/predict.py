# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentPosePredictor(DetectionPredictor):
    """Predictor for joint segmentation and pose estimation models.

    This predictor extends DetectionPredictor to handle both mask and keypoint outputs,
    producing Results objects containing bounding boxes, instance masks, and keypoints.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize SegmentPosePredictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment_pose"

    def postprocess(self, preds, img, orig_imgs):
        """Apply NMS and process segmentation + pose predictions for each image."""
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """Construct result objects from predictions."""
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """Construct a single result with boxes, masks, and keypoints."""
        if pred.shape[0] == 0:
            masks = None
            pred_kpts = None
        else:
            # Split prediction: [x1, y1, x2, y2, conf, cls, mask_coefficients..., keypoints...]
            nm = proto.shape[0]  # number of mask prototypes
            nk = self.model.kpt_shape[0] * self.model.kpt_shape[1]

            mask_coeff = pred[:, 6 : 6 + nm]
            kpt_data = pred[:, 6 + nm : 6 + nm + nk]

            # Process masks
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto, mask_coeff, pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto, mask_coeff, pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            # Filter predictions without valid masks
            if masks is not None:
                keep = masks.amax((-2, -1)) > 0
                if not all(keep):
                    pred, masks, kpt_data = pred[keep], masks[keep], kpt_data[keep]

            # Process keypoints
            pred_kpts = kpt_data.view(kpt_data.shape[0], *self.model.kpt_shape)
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            masks=masks,
            keypoints=pred_kpts,
        )
