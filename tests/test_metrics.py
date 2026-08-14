import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics import dice, hd95, volume_similarity, euler_diff, compute_metrics


def make_sphere(shape, center, radius):
    """ Create a binary 3D sphere mask."""
    grid = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((grid[0] - center[0])**2 + (grid[1] - center[1])**2 +(grid[2] - center[2])**2)
    return dist <= radius


# TESTS
#Test 1: identical spheres (Dice must be 1.0)
def test_identical():
    shape = (64, 64, 64)
    sphere = make_sphere(shape, center=(32, 32, 32), radius=10)
    # try individual metrics 
    result = dice(sphere, sphere)
    #result = hd95(sphere, sphere)
    #result = volume_similarity(sphere, sphere)
    #result = euler_diff(sphere, sphere)
    assert result == 1.0, f"Expected 1.0, got {result}"

# Test 2: empty prediction (Dice must be 0.0)
def test_empty():
    shape = (64, 64, 64)
    gt = make_sphere(shape, center=(32, 32, 32), radius=10)
    pred = np.zeros(shape, dtype=bool)
    result = dice(pred, gt)
    #result = hd95(pred, gt)
    #result = volume_similarity(pred, gt)
    #result = euler_diff(pred, gt)
    assert result == 0.0, f"Expected 0.0, got {result}"


# Test 3: both empty (Dice must be 1.0)
def test_both_empty():
    shape = (64, 64, 64)
    pred = np.zeros(shape, dtype=bool)
    gt = np.zeros(shape, dtype=bool)
    result = dice(pred, gt)
    #result = hd95(pred, gt)
    #result = volume_similarity(pred, gt)
    #result = euler_diff(pred, gt)
    assert result == 1.0, f"Expected 1.0, got {result}"

# Test 4: spheres partially overlap 
def test_partial():
    shape = (64, 64, 64)
    sphere1 = make_sphere(shape, center=(32, 32, 32), radius=10)
    sphere2 = make_sphere(shape, center=(32, 32, 42), radius=10)

    # compute expected dice manually
    intersection = np.sum(sphere1 & sphere2)
    expected = 2 * intersection / (np.sum(sphere1) + np.sum(sphere2))

    result = dice(sphere1, sphere2)
    #result = hd95(sphere1, sphere2)
    #result = volume_similarity(sphere1, sphere2)
    #result = euler_diff(sphere1, sphere2)
    # print(f"Expected {expected}, got {result}")
    assert abs(result - expected) < 1e-6


# ALL metrics
# identical spheres
def test_compute_metrics_identical():
    shape = (64, 64, 64)
    sphere = make_sphere(shape, center=(32, 32, 32), radius=10)
    pred = sphere.astype(np.uint8)
    gt= sphere.astype(np.uint8)

    voxel_spacing = (0.5, 0.5, 0.5)
    result = compute_metrics(pred, gt, voxel_spacing)

    assert result["mean"]["dice"] == 1.0
    assert result["mean"]["hd95"] == 0.0
    assert result["mean"]["volume_similarity"] == 1.0
    assert result["mean"]["euler_diff"] == 0.0

# empty prediction
def test_compute_metrics_empty_prediction():
    shape = (64, 64, 64)
    sphere = make_sphere(shape, center=(32, 32, 32), radius=10)

    pred = np.zeros(shape, dtype=np.uint8) # model predicts nothing
    gt = sphere.astype(np.uint8) # truth has tissue in class 1

    voxel_spacing = (0.5, 0.5, 0.5)
    result = compute_metrics(pred, gt, voxel_spacing)

    assert result[1]["dice"] == 0.0
    assert result[1]["hd95"] == 374.0
    assert result[1]["volume_similarity"] == 0.0

# partial overlap
def test_compute_metrics_partial_overlap():
    shape = (64, 64, 64)
    sphere1 = make_sphere(shape, center=(32, 32, 32), radius=10)
    sphere2 = make_sphere(shape, center=(32, 32, 42), radius=10)

    pred = sphere1.astype(np.uint8)
    gt = sphere2.astype(np.uint8)

    voxel_spacing = (0.5, 0.5, 0.5)
    result = compute_metrics(pred, gt, voxel_spacing)

    # dice must be between 0 and 1
    assert 0.0 < result[1]["dice"] < 1.0
    # hd95 must be greater than 0 
    assert result[1]["hd95"] > 0.0
    # both have same radius, vol_similarity should be 1
    assert result[1]["volume_similarity"] == 1.0
    # great that 0
    assert result[1]["euler_diff"] >= 0.0


if __name__ == "__main__":
    #test_identical() #passed
    #test_empty() # passed
    #test_both_empty() # passed
    #test_partial() # passed

    #test_compute_metrics_identical() # passed
    test_compute_metrics_empty_prediction() # passedd
    test_compute_metrics_partial_overlap() # passed