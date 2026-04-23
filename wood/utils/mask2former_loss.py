"""
MODIFIED: Mask2Former-style supervision for semantic segmentation targets.

Key fixes:
- background is treated as a normal semantic class target
- unmatched queries are supervised as no-object
- aux decoder outputs participate in the loss
- prediction masks are resized to the target spatial size before matching/loss
- matching cost is sanitized to avoid NaN/Inf crashes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Mask2FormerLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_cov=0.5,
        ignore_index=255,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        aux_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cov = lambda_cov
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, targets, cov_loss=None):
        total_loss, loss_dict = self._loss_single(outputs, targets)

        if isinstance(outputs, dict) and outputs.get("aux_outputs"):
            aux_class_loss = 0.0
            aux_mask_loss = 0.0
            aux_dice_loss = 0.0

            for aux_outputs in outputs["aux_outputs"]:
                aux_total, aux_dict = self._loss_single(aux_outputs, targets)
                total_loss = total_loss + self.aux_weight * aux_total
                aux_class_loss += aux_dict["class_loss"]
                aux_mask_loss += aux_dict["mask_loss"]
                aux_dice_loss += aux_dict["dice_loss"]

            num_aux = len(outputs["aux_outputs"])
            loss_dict["aux_class_loss"] = aux_class_loss / num_aux
            loss_dict["aux_mask_loss"] = aux_mask_loss / num_aux
            loss_dict["aux_dice_loss"] = aux_dice_loss / num_aux

        if cov_loss is not None:
            total_loss = total_loss + self.lambda_cov * cov_loss

        loss_dict["total_loss"] = total_loss.item()
        if cov_loss is not None:
            loss_dict["cov_loss"] = cov_loss.item() if torch.is_tensor(cov_loss) else cov_loss

        return total_loss, loss_dict

    def _loss_single(self, outputs, targets):
        pred_logits, pred_masks = self._unpack_outputs(outputs)
        target_instances = self._prepare_targets(targets)
        indices = self._hungarian_matching(pred_logits, pred_masks, target_instances)

        loss_ce = self._loss_labels(pred_logits, target_instances, indices)
        loss_mask = self._loss_masks(pred_masks, target_instances, indices)
        loss_dice = self._loss_dice(pred_masks, target_instances, indices)

        total_loss = (
            self.class_weight * loss_ce
            + self.mask_weight * loss_mask
            + self.dice_weight * loss_dice
        )

        return total_loss, {
            "class_loss": loss_ce.item() if torch.is_tensor(loss_ce) else loss_ce,
            "mask_loss": loss_mask.item() if torch.is_tensor(loss_mask) else loss_mask,
            "dice_loss": loss_dice.item() if torch.is_tensor(loss_dice) else loss_dice,
        }

    def _unpack_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs["pred_logits"], outputs["pred_masks"]
        return outputs.class_queries_logits, outputs.masks_queries_logits

    def _prepare_targets(self, semantic_labels):
        batch_size, height, width = semantic_labels.shape
        targets = []

        for batch_index in range(batch_size):
            label_map = semantic_labels[batch_index]
            valid_map = label_map != self.ignore_index

            class_labels = []
            class_masks = []

            # MODIFIED: include background as a normal semantic class target.
            for class_id in range(self.num_classes):
                mask = ((label_map == class_id) & valid_map).float()
                if mask.sum() > 0:
                    class_labels.append(class_id)
                    class_masks.append(mask)

            if class_labels:
                targets.append(
                    {
                        "labels": torch.tensor(class_labels, device=semantic_labels.device, dtype=torch.long),
                        "masks": torch.stack(class_masks),
                    }
                )
            else:
                targets.append(
                    {
                        "labels": torch.empty(0, device=semantic_labels.device, dtype=torch.long),
                        "masks": torch.zeros(0, height, width, device=semantic_labels.device),
                    }
                )

        return targets

    def _resize_pred_masks(self, pred_masks, target_hw):
        if pred_masks.shape[-2:] == target_hw:
            return pred_masks
        return F.interpolate(pred_masks, size=target_hw, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def _hungarian_matching(self, pred_logits, pred_masks, targets):
        indices = []

        for batch_index, target in enumerate(targets):
            out_prob = pred_logits[batch_index].softmax(-1)
            out_mask = pred_masks[batch_index]
            tgt_ids = target["labels"]
            tgt_masks = target["masks"]

            if tgt_ids.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                        torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                    )
                )
                continue

            out_mask = self._resize_pred_masks(out_mask.unsqueeze(0), tgt_masks.shape[-2:]).squeeze(0)
            out_mask_flat = out_mask.flatten(1).sigmoid()
            tgt_mask_flat = tgt_masks.flatten(1)

            cost_class = -out_prob[:, tgt_ids]

            numerator = 2 * torch.matmul(out_mask_flat, tgt_mask_flat.T)
            denominator = out_mask_flat.sum(-1, keepdim=True) + tgt_mask_flat.sum(-1, keepdim=True).T
            cost_dice = 1 - numerator / (denominator + 1e-8)

            cost = self.class_weight * cost_class + self.dice_weight * cost_dice
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)

            pred_idx, tgt_idx = linear_sum_assignment(cost.cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_logits.device),
                )
            )

        return indices

    def _loss_labels(self, pred_logits, targets, indices):
        batch_size, num_queries, _ = pred_logits.shape

        # MODIFIED: all queries are supervised; unmatched ones become no-object.
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.long,
            device=pred_logits.device,
        )

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[batch_index, src_idx] = targets[batch_index]["labels"][tgt_idx]

        empty_weight = torch.ones(self.num_classes + 1, device=pred_logits.device)
        empty_weight[-1] = self.no_object_weight

        return F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=empty_weight,
        )

    def _collect_matched_masks(self, pred_masks, targets, indices):
        matched_pred_masks = []
        matched_target_masks = []

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            target_masks = targets[batch_index]["masks"][tgt_idx]
            resized_pred_masks = self._resize_pred_masks(
                pred_masks[batch_index][src_idx].unsqueeze(0),
                target_masks.shape[-2:],
            ).squeeze(0)

            matched_pred_masks.append(resized_pred_masks)
            matched_target_masks.append(target_masks)

        if not matched_pred_masks:
            return None, None

        return torch.cat(matched_pred_masks), torch.cat(matched_target_masks)

    def _loss_masks(self, pred_masks, targets, indices):
        matched_pred_masks, matched_target_masks = self._collect_matched_masks(pred_masks, targets, indices)
        if matched_pred_masks is None:
            return pred_masks.sum() * 0.0

        return F.binary_cross_entropy_with_logits(
            matched_pred_masks,
            matched_target_masks,
            reduction="mean",
        )

    def _loss_dice(self, pred_masks, targets, indices):
        matched_pred_masks, matched_target_masks = self._collect_matched_masks(pred_masks, targets, indices)
        if matched_pred_masks is None:
            return pred_masks.sum() * 0.0

        matched_pred_masks = matched_pred_masks.sigmoid().flatten(1)
        matched_target_masks = matched_target_masks.flatten(1)

        numerator = 2 * (matched_pred_masks * matched_target_masks).sum(-1)
        denominator = matched_pred_masks.sum(-1) + matched_target_masks.sum(-1)
        return 1 - (numerator / (denominator + 1e-8)).mean()
