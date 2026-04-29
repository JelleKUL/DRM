import sys
import os
import json
import numpy as np
import trimesh
import drm

# Add mmdetection3d to path so imports work
sys.path.insert(0, os.path.abspath('../../mmdetection3d'))

from mmdet3d.apis import LidarDet3DInferencer


def create_inferencer(
    model_config,
    weights,
    device='cuda:0',
):
    """
    Load the model once and return the inferencer.

    Args:
        model_config : Path to the model config file
        weights      : Path to the checkpoint file
        device       : Inference device, e.g. 'cuda:0' or 'cpu'

    Returns:
        LidarDet3DInferencer: Ready-to-use inferencer instance
    """
    return LidarDet3DInferencer(
        model=model_config,
        weights=weights,
        device=device,
    )


def run_detection(
    inferencer,
    pcd_path,
    pred_score_thr=0.3,
    out_dir='outputs',
    save_pred=True,
    print_result=False,
):
    """
    Run detection on a point cloud using an existing inferencer.

    Args:
        inferencer    : A LidarDet3DInferencer instance from create_inferencer()
        pcd_path      : Path to the point cloud file (.bin or .pcd)
        pred_score_thr: Minimum confidence score for detections
        out_dir       : Directory to save outputs ('' to disable saving)
        save_pred     : Save prediction JSON files
        print_result  : Print predictions to stdout

    Returns:
        dict: Raw inference results
    """
    if not save_pred:
        out_dir = ''
    print(f'Running detection on {pcd_path} with score threshold {pred_score_thr}...')
    results = inferencer(
        inputs=dict(points=pcd_path),
        pred_score_thr=pred_score_thr,
        out_dir=out_dir,
        show=False,
        no_save_pred=not save_pred,
        print_result=print_result,
        num_workers=0,  # <-- prevents zombie dataloader workers
    )
    print('Detection completed.')
    if out_dir:
        print(f'Results saved to: {out_dir}')

    #return results


def load_gt_json_boxes_as_mesh(json_path, randomcolors = True):
    """
    Load detection results from a JSON file and convert them to a mesh format.

    Args:
        json_path: Path to the JSON file containing detection results
    """
    with open(json_path) as f:
        data = json.load(f)

    names = []
    points_list = []

    for box in data["boxes"]:
        names.append(box["id"])
        pts = np.array([[p["x"], p["y"], p["z"]] for p in box["boundingPoints"]])
        points_list.append(pts)

    # Colormap for labels
    label_colors = [
        [1, 0, 0, 0.5],  # red, alpha 0.5
        [0, 1, 0, 0.5],  # green
        [0, 0, 1, 0.5],  # blue
        [1, 1, 0, 0.5],  # yellow
        [1, 0, 1, 0.5],  # magenta
        [0, 1, 1, 0.5],  # cyan
    ]

    # Create list of meshes
    gt_bb_meshes = []
    i = 0
    for points in points_list:
        if(randomcolors):
            color = label_colors[i % len(label_colors)]
        else:
            color = [0, 0, 1, 0.5]  # blue, alpha 0.5

        mesh = trimesh.convex.convex_hull(points)
        mesh.visual.face_colors = color
        mesh.apply_transform(drm.UNITY2TRIMESH_T)
        gt_bb_meshes.append(mesh)
        i+=1

    return gt_bb_meshes

def load_detected_boxes_as_mesh(json_path, pred_score_thr=0.3, labelColors = True):
    """
    Load detection results from a JSON file and convert them to a mesh format.

    Args:
        json_path: Path to the JSON file containing detection results
        pred_score_thr: Minimum confidence score for detections to be included
        labelColors: Whether to use colors for different labels or a fixed color
    """
    with open(json_path) as f:
        data = json.load(f)

    # Colormap for labels
    label_colors = [
        [1, 0, 0, 0.5],  # red, alpha 0.5
        [0, 1, 0, 0.5],  # green
        [0, 0, 1, 0.5],  # blue
        [1, 1, 0, 0.5],  # yellow
        [1, 0, 1, 0.5],  # magenta
        [0, 1, 1, 0.5],  # cyan
    ]

    # Create list of meshes
    bb_meshes = []

    for label, score, box in zip(data["labels_3d"], data["scores_3d"], data["bboxes_3d"]):
        if score < pred_score_thr:
            continue

        center = np.array(box[:3])
        size = np.array(box[3:6])
        rotation_z = box[6]
        if(labelColors):
            color = label_colors[label % len(label_colors)]
        else:
            color = [1, 0, 0, 0.5]  # red, alpha 0.5
        
        mesh = drm.create_trimesh_box(center, size, rotation_z, color)
        bb_meshes.append(mesh)

    # Combine meshes for visualization
    return bb_meshes
