"""Small-target-aware assignment for NanoDet's Dynamic Soft Label Assigner.

This is a NanoDet adaptation of YOLO26 STAL, not a copy of Ultralytics'
TaskAlignedAssigner.  NanoDet-Plus uses DynamicSoftLabelAssigner (DSLA), so
the small-target candidate-geometry rule is applied before DSLA computes its
classification and IoU matching costs.
"""

import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .dsl_assigner import DynamicSoftLabelAssigner


class SmallTargetAwareDynamicSoftLabelAssigner(DynamicSoftLabelAssigner):
    """DSLA with STAL candidate geometry and optional one-to-one matching.

    For target assignment only, boxes with a width or height smaller than
    ``small_target_threshold`` are expanded about their centre to
    ``reference_size``. Regression targets always remain the original boxes.
    This prevents tiny boxes from having no feature-grid centre inside them.

    ``matching_mode='one_to_many'`` retains NanoDet-Plus DSLA's dynamic-k
    supervision. ``matching_mode='one_to_one'`` assigns at most one prediction
    to each ground-truth and at most one ground-truth to each prediction; it is
    intended for the NMS-free inference branch only.
    """

    def __init__(
        self,
        topk=13,
        iou_factor=3.0,
        ignore_iof_thr=-1,
        small_target_threshold=8.0,
        reference_size=16.0,
        matching_mode="one_to_many",
    ):
        super().__init__(topk, iou_factor, ignore_iof_thr)
        if matching_mode not in {"one_to_many", "one_to_one"}:
            raise ValueError("matching_mode must be 'one_to_many' or 'one_to_one'")
        if small_target_threshold <= 0 or reference_size <= 0:
            raise ValueError("STAL sizes must be positive")
        self.small_target_threshold = small_target_threshold
        self.reference_size = reference_size
        self.matching_mode = matching_mode

    def _assignment_boxes(self, gt_bboxes):
        """Return original boxes with only tiny-box candidate geometry enlarged."""
        assignment_boxes = gt_bboxes.clone()
        wh = assignment_boxes[:, 2:] - assignment_boxes[:, :2]
        small = (wh[:, 0] < self.small_target_threshold) | (
            wh[:, 1] < self.small_target_threshold
        )
        if small.any():
            centers = (assignment_boxes[small, :2] + assignment_boxes[small, 2:]) * 0.5
            half = assignment_boxes.new_full((small.sum(), 2), self.reference_size * 0.5)
            assignment_boxes[small, :2] = centers - half
            assignment_boxes[small, 2:] = centers + half
        return assignment_boxes

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        inf = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_zeros(num_bboxes, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            labels = (
                None
                if gt_labels is None
                else decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            )
            return AssignResult(num_gt, assigned_gt_inds, decoded_bboxes.new_zeros(num_bboxes), labels=labels)

        assignment_boxes = self._assignment_boxes(gt_bboxes)
        prior_center = priors[:, :2]
        deltas = torch.cat(
            [
                prior_center[:, None] - assignment_boxes[:, :2],
                assignment_boxes[:, 2:] - prior_center[:, None],
            ],
            dim=-1,
        )
        valid_mask = (deltas.min(dim=-1).values > 0).any(dim=1)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]

        if valid_decoded_bbox.numel() == 0:
            labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, decoded_bboxes.new_zeros(num_bboxes), labels=labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-7)
        gt_onehot = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float()
        soft_label = gt_onehot.unsqueeze(0) * pairwise_ious[..., None]
        score_delta = soft_label - valid_pred_scores.unsqueeze(1).sigmoid()
        cls_cost = (
            F.binary_cross_entropy_with_logits(
                valid_pred_scores.unsqueeze(1).expand(-1, num_gt, -1),
                soft_label,
                reduction="none",
            )
            * score_delta.abs().pow(2.0)
        ).sum(dim=-1)
        cost_matrix = cls_cost + iou_cost * self.iou_factor

        if self.matching_mode == "one_to_one":
            matched_ious, matched_gt_inds, matched_prior_inds = self._one_to_one_matching(
                cost_matrix, pairwise_ious
            )
            max_overlaps = decoded_bboxes.new_full((num_bboxes,), -inf)
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            if matched_prior_inds.numel() > 0:
                full_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)[matched_prior_inds]
                assigned_gt_inds[full_indices] = matched_gt_inds + 1
                max_overlaps[full_indices] = matched_ious
                assigned_labels[full_indices] = gt_labels[matched_gt_inds].long()
        else:
            matched_ious, matched_gt_inds = self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask
            )
            assigned_gt_inds[valid_mask] = matched_gt_inds + 1
            max_overlaps = decoded_bboxes.new_full((num_bboxes,), -inf)
            max_overlaps[valid_mask] = matched_ious
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                decoded_bboxes[valid_mask], gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            assigned_gt_inds[valid_indices[ignore_idxs]] = -1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def _one_to_one_matching(self, cost, pairwise_ious):
        """Greedily select unique low-cost prior/GT pairs.

        The selection is globally sorted by cost, therefore no selected prior is
        reused and no ground truth receives more than one positive. It avoids a
        SciPy dependency while giving the o2o branch genuine one-to-one labels.
        """
        num_priors, num_gt = cost.shape
        order = cost.flatten().argsort()
        prior_used = torch.zeros(num_priors, dtype=torch.bool, device=cost.device)
        gt_used = torch.zeros(num_gt, dtype=torch.bool, device=cost.device)
        selected_priors, selected_gts = [], []
        for flat_index in order:
            prior_index = torch.div(flat_index, num_gt, rounding_mode="floor")
            gt_index = flat_index.remainder(num_gt)
            if not prior_used[prior_index] and not gt_used[gt_index]:
                prior_used[prior_index] = True
                gt_used[gt_index] = True
                selected_priors.append(prior_index)
                selected_gts.append(gt_index)
                if gt_used.all():
                    break

        if not selected_priors:
            empty = cost.new_zeros(0)
            return empty, empty.long(), empty.long()

        prior_indices = torch.stack(selected_priors)
        gt_indices = torch.stack(selected_gts)
        return pairwise_ious[prior_indices, gt_indices], gt_indices, prior_indices
