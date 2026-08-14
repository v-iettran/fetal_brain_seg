import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.ndimage import label
from skimage.measure import euler_number

HD95_PENALTY = 374.0  # max possible distance in mm


# INDIVIDUAL METRICS

def dice(pred, gt):
    """
    DICE measures overlap: 0 = no overlap, 1 = perfect. Our primary metric.
    """
    intersection = np.sum(pred & gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    elif pred_sum == 0 or gt_sum == 0:
        return 0.0
    else:
        # 2 × (voxels in both) / (voxels in A + voxels in B)
        return float(2 * intersection / (pred_sum + gt_sum)) 


def hd95(pred, gt, voxel_spacing):
    """
    HD95 measures boundary error (the 95th percentile of the distance between the boundaries of two masks - perdicted and truth).
    """
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 or gt_sum == 0:
        return HD95_PENALTY

    # step 1: extract surface voxels
    pred_surface = pred & ~binary_erosion(pred)
    gt_surface   = gt   & ~binary_erosion(gt)

    # step 2: convert voxel coordinates to mm
    spacing = np.array(voxel_spacing)
    pred_pts = np.argwhere(pred_surface) * spacing
    gt_pts   = np.argwhere(gt_surface)   * spacing

    # step 3: find nearest neighbour distances in both directions
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)

    # step 4: percentile 95 over all distances
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    return float(np.percentile(all_distances, 95))


def volume_similarity(pred, gt):
    """
    VOLUME SIMILARITY measures if the total volume is right (ignoring the location of that volume)
    """
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    elif pred_sum == 0 or gt_sum == 0:
        return 0.0
    else:
        # VS = 1 - |volumen_pred - volumen_gt| / (volumen_pred + volumen_gt)
        return float(1 - abs(pred_sum - gt_sum) / (pred_sum + gt_sum)) 



def euler_diff(pred, gt):
    """
    EULER CHARACTERISTIC DIFFERENCE measures the topological error (if the predicted structure have the right number of holes and connected pieces)
    """
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    elif pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    # euler_diff = |euler(pred) - euler(gt)|
    euler_pred = euler_number(pred, connectivity=3)
    euler_gt   = euler_number(gt,   connectivity=3)

    return float(abs(euler_pred - euler_gt))



# Combine metrics for one class
def _metrics_for_class(pred, gt, voxel_spacing):
    """
    Compute metrics for a single binary mask (only one tissue)
    """
    return {
        "dice": dice(pred, gt),
        "hd95": hd95(pred, gt, voxel_spacing),
        "volume_similarity": volume_similarity(pred, gt),
        "euler_diff": euler_diff(pred, gt),
    }

# main - combine metrics for all classes 
def compute_metrics(prediction, ground_truth, voxel_spacing):
    """
    Compute segmentation metrics for each tissue class
    """

    classes = [1, 2, 3, 4, 5, 6, 7]
    results = {}

    for c in classes:
        pred_c = (prediction == c)
        gt_c   = (ground_truth == c)
        results[c] = _metrics_for_class(pred_c, gt_c, voxel_spacing)

    # mean across all 7 classes
    results["mean"] = {
        metric: np.mean([results[c][metric] for c in classes])
        for metric in results[1]
    }

    return results