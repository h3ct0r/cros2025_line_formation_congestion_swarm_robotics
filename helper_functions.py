import numpy as np


def normalize_vector(vec):
    """Normalizes a 2D vector. Returns a zero vector if the input is zero."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 0.0])
    return vec / norm


def euclidean_distance(pos1, pos2):
    """Calculates the Euclidean distance between two 2D points."""
    return np.linalg.norm(pos1 - pos2)


def normalize_angle_rad(angle_rad):
    """
    Normalizes an angle in radians to the range [-π, π).

    Args:
        angle_rad (float): The angle in radians to normalize.

    Returns:
        float: The normalized angle in radians within the range [-π, π).
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
