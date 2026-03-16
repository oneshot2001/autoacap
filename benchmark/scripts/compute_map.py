#!/usr/bin/env python3
"""
Compute mean Average Precision (mAP@0.50) for person detection.
Compares ACAP detections against ground truth labels.

Usage:
    python3 compute_map.py --detections /tmp/metrics.json --ground-truth ground_truth/detections.json

Input formats:
    Ground truth (COCO-like):
    {
        "frames": [
            {
                "frame_id": 0,
                "detections": [{"bbox": [x, y, w, h], "class": "person"}]
            }
        ]
    }

    ACAP detections (from camera):
    {
        "frames": [
            {
                "frame_id": 0,
                "detections": [{"bbox": [x, y, w, h], "confidence": 0.85, "class": "person"}]
            }
        ]
    }

Output:
    {"mAP_50": 0.623}
"""

import argparse
import json
import sys


def iou(box_a, box_b):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to [x1, y1, x2, y2]
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_ap(gt_frames, det_frames, iou_threshold=0.5):
    """Compute Average Precision at a given IoU threshold."""
    # Collect all detections with their confidence
    all_detections = []
    total_gt = 0

    # Index ground truth by frame_id
    gt_by_frame = {}
    for frame in gt_frames:
        fid = frame["frame_id"]
        gt_boxes = [d["bbox"] for d in frame.get("detections", []) if d.get("class") == "person"]
        gt_by_frame[fid] = gt_boxes
        total_gt += len(gt_boxes)

    if total_gt == 0:
        return 0.0

    # Collect all detections
    for frame in det_frames:
        fid = frame["frame_id"]
        for det in frame.get("detections", []):
            if det.get("class") == "person":
                all_detections.append({
                    "frame_id": fid,
                    "bbox": det["bbox"],
                    "confidence": det.get("confidence", 0.0),
                })

    # Sort by confidence (descending)
    all_detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Track which GT boxes have been matched
    gt_matched = {fid: [False] * len(boxes) for fid, boxes in gt_by_frame.items()}

    tp = []
    fp = []

    for det in all_detections:
        fid = det["frame_id"]
        gt_boxes = gt_by_frame.get(fid, [])

        best_iou = 0.0
        best_idx = -1

        for idx, gt_box in enumerate(gt_boxes):
            curr_iou = iou(det["bbox"], gt_box)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0 and not gt_matched[fid][best_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[fid][best_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Compute precision-recall curve
    tp_cumsum = []
    fp_cumsum = []
    tp_sum = 0
    fp_sum = 0

    for t, f in zip(tp, fp):
        tp_sum += t
        fp_sum += f
        tp_cumsum.append(tp_sum)
        fp_cumsum.append(fp_sum)

    precisions = []
    recalls = []

    for tp_c, fp_c in zip(tp_cumsum, fp_cumsum):
        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        recall = tp_c / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # 11-point interpolation
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_interp = 0.0
        for p, r in zip(precisions, recalls):
            if r >= t:
                p_interp = max(p_interp, p)
        ap += p_interp / 11.0

    return ap


def main():
    parser = argparse.ArgumentParser(description="Compute mAP@0.50 for person detection")
    parser.add_argument("--detections", required=True, help="Path to ACAP detection results JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth labels JSON")
    args = parser.parse_args()

    try:
        with open(args.ground_truth) as f:
            gt_data = json.load(f)
        with open(args.detections) as f:
            det_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"mAP_50": 0.0, "error": str(e)}))
        sys.exit(0)

    gt_frames = gt_data.get("frames", [])
    det_frames = det_data.get("frames", [])

    if not gt_frames:
        print(json.dumps({"mAP_50": 0.0, "error": "no ground truth frames"}))
        sys.exit(0)

    map_50 = compute_ap(gt_frames, det_frames, iou_threshold=0.5)

    print(json.dumps({"mAP_50": round(map_50, 4)}))


if __name__ == "__main__":
    main()
