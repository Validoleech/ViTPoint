from datasets.utils.synthetic_dataset_config import PersistentSynthConfig
import numpy as np
import cv2 as cv
import numpy
from typing import Tuple, List, Dict, Any, Union

# Global variable to store the random state, allowing for reproducible results.
random_state: Union[np.random.RandomState, None] = None

cfg = PersistentSynthConfig()

def set_random_state(state: np.random.RandomState) -> None:
    """Sets the global random state for reproducibility.

    Args:
        state: An instance of numpy.random.RandomState to be used globally.
    """
    global random_state
    random_state = state


def get_random_color(background_color: int) -> int:
    """Generates a random grayscale scalar color with sufficient contrast to the background.

    Ensures the generated color differs from the background by at least 30 to guarantee visibility.

    Args:
        background_color: The integer scalar value of the background color (0-255).

    Returns:
        A random grayscale scalar color (0-255).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    color: int = random_state.randint(256)
    if abs(color - background_color) < 30:  # Not enough contrast
        color = (color + 128) % 256  # Shift color to ensure contrast
    return color


def get_different_color(previous_colors: np.ndarray, min_dist: int = 50, max_count: int = 20) -> int:
    """Generates a color that contrasts with a list of previous colors.

    Iteratively generates random colors until one is found that is sufficiently
    different from all `previous_colors`, or `max_count` iterations are reached.

    Args:
        previous_colors: A NumPy array of integer scalar values representing previous colors.
        min_dist: The minimum required absolute difference between the new color and any previous color.
        max_count: The maximum number of attempts to find a contrasting color.

    Returns:
        An integer scalar color (0-255) that contrasts with `previous_colors`.
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    color: int = random_state.randint(256)
    count: int = 0
    # Loop until color is sufficiently different or max attempts reached
    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = random_state.randint(256)
    return color


def add_salt_and_pepper(img: np.ndarray) -> np.ndarray:
    """Adds salt-and-pepper noise to a grayscale image.

    Randomly sets pixels to black (0) or white (255) based on thresholds,
    then applies a blur to smooth the noise.

    Args:
        img: The input grayscale image as a NumPy array (modified in place).

    Returns:
        An empty NumPy array of shape (0, 2) and dtype int32.
    """
    # Generate a uniform random noise mask
    noise: np.ndarray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(noise, 0, 255)

    # Apply salt (white) and pepper (black) noise based on thresholds
    black: np.ndarray = noise < 30
    white: np.ndarray = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0

    # Apply a blur to smooth the noise
    cv.blur(img, (5, 5), img)
    return np.empty((0, 2), dtype=np.int32)


def generate_background(size: Tuple[int, int] = (960, 1280), nb_blobs: int = 100,
                        min_rad_ratio: float = 0.01, max_rad_ratio: float = 0.05,
                        min_kernel_size: int = 50, max_kernel_size: int = 300) -> np.ndarray:
    """Generates a customized background image with random blobs and blurring.

    Initializes an image with random noise, thresholds it, then draws multiple
    randomly sized and positioned circles (blobs) with contrasting colors,
    and finally applies a large blur.

    Args:
        size: The (height, width) of the generated image.
        nb_blobs: The number of circles to draw on the background.
        min_rad_ratio: Minimum radius of blobs as a ratio of the max image dimension.
        max_rad_ratio: Maximum radius of blobs as a ratio of the max image dimension.
        min_kernel_size: The minimum side length for the blurring kernel.
        max_kernel_size: The maximum side length for the blurring kernel.

    Returns:
        The generated grayscale background image as a NumPy array.
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    img: np.ndarray = np.zeros(size, dtype=np.uint8)
    dim: int = max(size)

    # Initialize image with random values and threshold to binary
    cv.randu(img, 0, 255)
    cv.threshold(img, random_state.randint(256), 255, cv.THRESH_BINARY, img)
    background_color: int = int(np.mean(img))

    # Generate random blob positions
    blobs: np.ndarray = np.concatenate([random_state.randint(0, size[1], size=(nb_blobs, 1)),
                                        random_state.randint(0, size[0], size=(nb_blobs, 1))],
                                       axis=1)

    # Draw blobs with random contrasting colors and sizes
    for i in range(nb_blobs):
        col: int = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  random_state.randint(int(dim * min_rad_ratio),
                                       int(dim * max_rad_ratio)),
                  col, -1)  # -1 for filled circle

    # Apply a large blur to create a smooth background
    kernel_size: int = random_state.randint(min_kernel_size, max_kernel_size)
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def generate_custom_background(size: Tuple[int, int], background_color: int, nb_blobs: int = 3000,
                               kernel_boundaries: Tuple[int, int] = (50, 100)) -> np.ndarray:
    """Generates a customized background image, typically used to fill shapes.

    Initializes an image with a random color contrasting with the `background_color`,
    then draws many small random circles (blobs), and applies blurring.

    Args:
        size: The (height, width) of the generated image.
        background_color: The average color of the main background image to ensure contrast.
        nb_blobs: The number of small circles to draw.
        kernel_boundaries: A tuple (min_size, max_size) for the blurring kernel's dimensions.

    Returns:
        The generated grayscale custom background image as a NumPy array.
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    img: np.ndarray = np.zeros(size, dtype=np.uint8)
    # Fill with a contrasting base color
    img = img + get_random_color(background_color)

    # Generate random blob positions
    blobs: np.ndarray = np.concatenate([random_state.randint(0, size[1], size=(nb_blobs, 1)),
                                        random_state.randint(0, size[0], size=(nb_blobs, 1))],
                                       axis=1)

    # Draw many small blobs
    for i in range(nb_blobs):
        col: int = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  random_state.randint(20), col, -1) # Fixed small max radius

    # Apply blur
    kernel_size: int = random_state.randint(
        kernel_boundaries[0], kernel_boundaries[1])
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def final_blur(img: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> None:
    """Applies a final Gaussian blur to the image.

    Modifies the input image in place.

    Args:
        img: The input image as a NumPy array (modified in place).
        kernel_size: The size of the Gaussian blurring kernel (width, height).
    """
    cv.GaussianBlur(img, kernel_size, 0, img)


def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray, dim: int) -> np.ndarray:
    """Checks if three 2D or 3D points are in counter-clockwise (CCW) order.

    Calculates the cross product of vectors (B-A) and (C-A) to determine orientation.

    Args:
        A: First point(s) as a NumPy array.
        B: Second point(s) as a NumPy array.
        C: Third point(s) as a NumPy array.
        dim: The dimension of the points (2 for 2D, 3 for 3D).

    Returns:
        A boolean NumPy array indicating if points are in CCW order.
    """
    if dim == 2:  # Only 2 dimensions
        return ((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
                > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
    else:  # dim should be equal to 3 for batched intersection checks
        return ((C[:, 1, :] - A[:, 1, :])
                * (B[:, 0, :] - A[:, 0, :])
                > (B[:, 1, :] - A[:, 1, :])
                * (C[:, 0, :] - A[:, 0, :]))


def intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, dim: int) -> bool:
    """Checks if two line segments (AB and CD) intersect.

    Uses the counter-clockwise (CCW) orientation test.

    Args:
        A: Start point(s) of the first segment.
        B: End point(s) of the first segment.
        C: Start point(s) of the second segment.
        D: End point(s) of the second segment.
        dim: The dimension of the points (2 for 2D, 3 for batched 3D).

    Returns:
        True if the line segments intersect, False otherwise.
    """
    # Segments AB and CD intersect if C and D are on opposite sides of AB,
    # AND A and B are on opposite sides of CD.
    return np.any((ccw(A, C, D, dim) != ccw(B, C, D, dim)) &
                  (ccw(A, B, C, dim) != ccw(A, B, D, dim)))


def keep_points_inside(points: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Filters an array of points, keeping only those whose coordinates are within image boundaries.

    Args:
        points: A NumPy array of points (N, 2), where N is the number of points.
        size: A tuple (height, width) representing the image dimensions.

    Returns:
        A NumPy array containing only the points that are inside the image boundaries.
    """
    # Create a mask for points within x (width) and y (height) boundaries
    mask: np.ndarray = (points[:, 0] >= 0) & (points[:, 0] < size[1]) & \
                       (points[:, 1] >= 0) & (points[:, 1] < size[0])
    return points[mask, :]


def draw_lines(img: np.ndarray, nb_min_lines: int = 30, nb_max_lines: int = 50) -> np.ndarray:
    """Draws random lines on an image and returns the positions of their endpoints.

    Ensures that newly drawn lines do not significantly overlap with existing ones.

    Args:
        img: The input image as a NumPy array (modified in place).
        nb_min_lines: The minimum number of lines to attempt to draw.
        nb_max_lines: The maximum number of lines to attempt to draw.

    Returns:
        A NumPy array of the endpoints of the drawn lines (N, 2), where N is the total number of endpoints.
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        nb_min_lines *= 3
        nb_max_lines *= 3
    num_lines: int = random_state.randint(nb_min_lines, nb_max_lines + 1)
    # Stores (x1, y1, x2, y2) for drawn lines
    segments: np.ndarray = np.empty((0, 4), dtype=np.int32)
    points: np.ndarray = np.empty(
        (0, 2), dtype=np.int32)  # Stores all endpoints
    background_color: int = int(np.mean(img))
    min_dim: int = min(img.shape)

    for _ in range(num_lines):
        x1: int = random_state.randint(img.shape[1])
        y1: int = random_state.randint(img.shape[0])
        p1: np.ndarray = np.array([[x1, y1]])
        x2: int = random_state.randint(img.shape[1])
        y2: int = random_state.randint(img.shape[0])
        p2: np.ndarray = np.array([[x2, y2]])

        # Check that there is no significant overlap with existing segments
        if segments.shape[0] > 0 and intersect(segments[:, 0:2], segments[:, 2:4], p1, p2, 2):
            continue

        # Add the new segment and its endpoints
        segments = np.concatenate(
            [segments, np.array([[x1, y1, x2, y2]])], axis=0)
        col: int = get_random_color(background_color)
        thickness: int = random_state.randint(
            int(min_dim * 0.01), int(min_dim * 0.02))
        cv.line(img, (x1, y1), (x2, y2), col, thickness)
        points = np.concatenate(
            [points, np.array([[x1, y1], [x2, y2]])], axis=0)
    return points


def draw_polygon(img: np.ndarray, max_sides: int = 8) -> np.ndarray:
    """Draws a filled polygon with a random number of corners on an image and returns the corner points.

    Randomly selects a center and radius, then samples points on a circle.
    Filters points that are too close or create very flat angles.

    Args:
        img: The input image as a NumPy array (modified in place).
        max_sides: The maximum number of sides (and corners) the polygon can have + 1.

    Returns:
        A NumPy array of the corner points of the drawn polygon (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    num_corners: int = random_state.randint(3, max_sides)
    min_dim: int = min(img.shape[0], img.shape[1])
    # Random radius, ensuring a minimum size
    rad: float = max(random_state.rand() * min_dim / 2, min_dim / 10)
    # Random center of the circle within image bounds, respecting radius
    x: int = random_state.randint(int(rad), img.shape[1] - int(rad))
    y: int = random_state.randint(int(rad), img.shape[0] - int(rad))

    # Sample points inside the circle, ensuring a minimum distance from center
    slices: np.ndarray = np.linspace(0, 2 * numpy.pi, num_corners + 1)
    angles: List[float] = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                           for i in range(num_corners)]
    points: np.ndarray = np.array([[int(x + max(random_state.rand(), 0.4) * rad * numpy.cos(a)),
                                    int(y + max(random_state.rand(), 0.4) * rad * numpy.sin(a))]
                                   for a in angles])

    # Filter points that are too close to each other
    norms: List[float] = [np.linalg.norm(points[(i-1) % num_corners, :] - points[i, :])
                          for i in range(num_corners)]
    mask_dist: np.ndarray = np.array(norms) > 0.01
    points = points[mask_dist, :]
    num_corners = points.shape[0]

    # Filter points that form angles that are too flat (less than 60 degrees)
    if num_corners >= 3:
        corner_angles: List[float] = [angle_between_vectors(points[(i-1) % num_corners, :] -
                                                            points[i, :],
                                                            points[(i+1) % num_corners, :] -
                                                            points[i, :])
                                      for i in range(num_corners)]
        mask_angle: np.ndarray = np.array(corner_angles) < (
            2 * numpy.pi / 3)  # Angle must be < 120 deg (2*pi/3 rad)
        points = points[mask_angle, :]
        num_corners = points.shape[0]

    # If not enough valid corners, retry drawing a polygon
    if num_corners < 3:
        return draw_polygon(img, max_sides)

    # Reshape corners for OpenCV and fill the polygon
    corners: np.ndarray = points.reshape((-1, 1, 2))
    col: int = get_random_color(int(np.mean(img)))
    cv.fillPoly(img, [corners], col)
    return points


def overlap(center: np.ndarray, rad: float, centers: List[np.ndarray], rads: List[float]) -> bool:
    """Checks that a new circle does not significantly overlap with existing circles.

    Considers overlap based on distance between centers and radii.

    Args:
        center: The (x, y) coordinates of the new circle's center.
        rad: The radius of the new circle.
        centers: A list of (x, y) coordinates for existing circle centers.
        rads: A list of radii for existing circles.

    Returns:
        True if the new circle overlaps with any existing circles, False otherwise.
    """
    flag: bool = False
    for i in range(len(rads)):
        # Overlap if distance between centers + min(radii) < max(radii)
        # This condition is typically for one circle being completely inside another or touching.
        # For general overlap, it should be np.linalg.norm(center - centers[i]) < (rad + rads[i]).
        # The original code's logic is a bit specific and might allow some partial overlaps.
        if np.linalg.norm(center - centers[i]) + min(rad, rads[i]) < max(rad, rads[i]):
            flag = True
            break
    return flag


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes the angle (in radians) between two 2D vectors.

    Args:
        v1: The first 2D vector as a NumPy array.
        v2: The second 2D vector as a NumPy array.

    Returns:
        The angle in radians between the two vectors.
    """
    # Normalize vectors to unit vectors
    v1_u: np.ndarray = v1 / np.linalg.norm(v1)
    v2_u: np.ndarray = v2 / np.linalg.norm(v2)
    # Compute arccos of dot product, clipping to avoid floating point errors outside [-1, 1]
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def draw_multiple_polygons(img: np.ndarray, max_sides: int = 8, nb_polygons: int = 20,
                           **extra: Any) -> np.ndarray:
    """Draws multiple filled polygons on an image and returns their corner points.

    Each polygon is filled with a custom background generated by `generate_custom_background`.
    Checks for overlaps between polygons to prevent drawing on top of existing ones.

    Args:
        img: The input image as a NumPy array (modified in place).
        max_sides: The maximum number of sides (and corners) a polygon can have + 1.
        nb_polygons: The maximum number of polygons to attempt to draw.
        **extra: Additional keyword arguments passed directly to `generate_custom_background`.

    Returns:
        A NumPy array of the corner points of all drawn polygons (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        nb_polygons *= 3
    # Stores segments of existing polygons (x1, y1, x2, y2)
    segments: np.ndarray = np.empty((0, 4), dtype=np.int32)
    # Stores centers of drawn polygons for overlap check
    centers: List[np.ndarray] = []
    rads: List[float] = []  # Stores radii of drawn polygons for overlap check
    points: np.ndarray = np.empty(
        (0, 2), dtype=np.int32)  # Stores all corner points
    background_color: int = int(np.mean(img))

    for _ in range(nb_polygons):
        # Generate polygon parameters similar to draw_polygon
        num_corners: int = random_state.randint(3, max_sides)
        min_dim: int = min(img.shape[0], img.shape[1])
        rad: float = max(random_state.rand() * min_dim / 2, min_dim / 10)
        x: int = random_state.randint(
            int(rad), img.shape[1] - int(rad))  # Center of a circle
        y: int = random_state.randint(int(rad), img.shape[0] - int(rad))

        # Sample num_corners points inside the circle
        slices: np.ndarray = np.linspace(0, 2 * numpy.pi, num_corners + 1)
        angles: List[float] = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                               for i in range(num_corners)]
        new_points: np.ndarray = np.array([[int(x + max(random_state.rand(), 0.4) * rad * numpy.cos(a)),
                                            int(y + max(random_state.rand(), 0.4) * rad * numpy.sin(a))]
                                           for a in angles])

        # Filter points that are too close or have an angle too flat
        norms: List[float] = [np.linalg.norm(new_points[(i-1) % num_corners, :] - new_points[i, :])
                              for i in range(num_corners)]
        mask_dist: np.ndarray = np.array(norms) > 0.01
        new_points = new_points[mask_dist, :]
        num_corners = new_points.shape[0]

        if num_corners >= 3:
            corner_angles: List[float] = [angle_between_vectors(new_points[(i-1) % num_corners, :] -
                                                                new_points[i, :],
                                                                new_points[(i+1) % num_corners, :] -
                                                                new_points[i, :])
                                          for i in range(num_corners)]
            mask_angle: np.ndarray = np.array(
                corner_angles) < (2 * numpy.pi / 3)
            new_points = new_points[mask_angle, :]
            num_corners = new_points.shape[0]

        if num_corners < 3:  # Not enough valid corners, skip this polygon
            continue

        # Prepare new polygon segments for intersection check
        new_segments: np.ndarray = np.zeros(
            (1, 4, num_corners), dtype=np.int32)
        new_segments[0, 0, :] = [new_points[i][0] for i in range(num_corners)]
        new_segments[0, 1, :] = [new_points[i][1] for i in range(num_corners)]
        new_segments[0, 2, :] = [new_points[(i+1) % num_corners][0]
                                 for i in range(num_corners)]
        new_segments[0, 3, :] = [new_points[(i+1) % num_corners][1]
                                 for i in range(num_corners)]

        # Check for overlap with pre-existing shapes
        # The intersection check uses dim=3 for batched segments (all segments of new polygon vs all previous segments)
        if (segments.shape[0] > 0 and intersect(segments[:, 0:2, None], segments[:, 2:4, None],
                                                new_segments[:, 0:2,
                                                             :], new_segments[:, 2:4, :],
                                                3)) or overlap(np.array([x, y]), rad, centers, rads):
            continue

        # Add current polygon's center and radius to lists for future overlap checks
        centers.append(np.array([x, y]))
        rads.append(rad)
        # Reshape new_segments for concatenation with previous segments
        new_segments = np.reshape(np.swapaxes(new_segments, 0, 2), (-1, 4))
        segments = np.concatenate([segments, new_segments], axis=0)

        # Color the polygon with a custom background generated on the fly
        corners: np.ndarray = new_points.reshape((-1, 1, 2))
        # Create a mask for the polygon area
        mask: np.ndarray = np.zeros(img.shape, np.uint8)
        custom_background: np.ndarray = generate_custom_background(
            img.shape, background_color, **extra)
        cv.fillPoly(mask, [corners], 255)  # Draw polygon onto the mask
        locs: Tuple[np.ndarray, np.ndarray] = np.where(
            mask != 0)  # Get coordinates where mask is non-zero
        img[locs[0], locs[1]] = custom_background[locs[0],
                                                  # Apply custom background
                                                  locs[1]]
        points = np.concatenate([points, new_points], axis=0)
    return points


def draw_ellipses(img: np.ndarray, nb_ellipses: int = 20) -> np.ndarray:
    """Draws multiple filled ellipses on an image.

    Checks for overlaps to prevent drawing on top of existing ellipses.

    Args:
        img: The input image as a NumPy array (modified in place).
        nb_ellipses: The maximum number of ellipses to attempt to draw.

    Returns:
        An empty NumPy array of shape (0, 2) and dtype int32 (no specific points are tracked).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    # Stores centers of drawn ellipses
    centers: np.ndarray = np.empty((0, 2), dtype=np.int32)
    # Stores max radii of drawn ellipses
    rads: np.ndarray = np.empty((0, 1), dtype=np.int32)
    min_dim: float = min(img.shape[0], img.shape[1]) / 4
    background_color: int = int(np.mean(img))

    for _ in range(nb_ellipses):
        # Generate random axes and center for the ellipse
        ax: int = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay: int = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad: int = max(ax, ay)
        x: int = random_state.randint(
            max_rad, img.shape[1] - max_rad)  # Center x
        y: int = random_state.randint(
            max_rad, img.shape[0] - max_rad)  # Center y
        new_center: np.ndarray = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        # Overlap check based on distance between centers and radii sum.
        # This checks if the new ellipse *overlaps* with any previous ellipse.
        # The condition (max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads))
        # implies overlap if new_rad + prev_rad > dist.
        if centers.shape[0] > 0:
            diff: np.ndarray = centers - new_center
            if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
                continue

        # Add current ellipse's center and radius to lists
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        # Draw the filled ellipse
        col: int = get_random_color(background_color)
        angle: float = random_state.rand() * 90  # Random rotation angle
        cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360,
                   col, -1)  # -1 for filled ellipse
    return np.empty((0, 2), dtype=np.int32)


def draw_star(img: np.ndarray, nb_branches: int = 6) -> np.ndarray:
    """Draws a star shape on an image and outputs its interest points.

    The star is drawn by connecting a central point to several points
    randomly placed on a circle.

    Args:
        img: The input image as a NumPy array (modified in place).
        nb_branches: The maximum number of branches (points on the outer circle) for the star + 1.

    Returns:
        A NumPy array of the interest points (center and branch ends) of the drawn star (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        nb_branches *= 3
    num_branches: int = random_state.randint(3, nb_branches)
    min_dim: int = min(img.shape[0], img.shape[1])
    thickness: int = random_state.randint(
        int(min_dim * 0.01), int(min_dim * 0.02))
    rad: float = max(random_state.rand() * min_dim / 2, min_dim / 5)

    # Select the center of a circle for the star
    x: int = random_state.randint(int(rad), img.shape[1] - int(rad))
    y: int = random_state.randint(int(rad), img.shape[0] - int(rad))

    # Sample points on the circle for the branches, ensuring a minimum distance from center
    slices: np.ndarray = np.linspace(0, 2 * numpy.pi, num_branches + 1)
    angles: List[float] = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                           for i in range(num_branches)]
    points: np.ndarray = np.array([[int(x + max(random_state.rand(), 0.3) * rad * numpy.cos(a)),
                                    int(y + max(random_state.rand(), 0.3) * rad * numpy.sin(a))]
                                   for a in angles])
    # Add the center point as the first point
    points = np.concatenate(([[x, y]], points), axis=0)

    background_color: int = int(np.mean(img))

    # Draw lines from the center to each branch point
    for i in range(1, num_branches + 1):
        col: int = get_random_color(background_color)
        cv.line(img, (points[0][0], points[0][1]),
                (points[i][0], points[i][1]),
                col, thickness)
    return points


def draw_multiple_stars(img: np.ndarray, nb_stars: int = 25,
                        max_branches: int = 8) -> np.ndarray:
    """
    Draw many star shapes (random number of branches, random size / colour).
    Returns all branch-end points plus the centre of each star.
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        nb_branches *= 3
    min_dim = min(img.shape)
    thickness_rng = (int(min_dim * 0.005), int(min_dim * 0.02))
    background_color = int(np.mean(img))

    all_pts = []
    for _ in range(nb_stars):
        # random geometry
        num_br = random_state.randint(3, max_branches + 1)
        rad = max(random_state.rand() * min_dim / 4, min_dim / 15)
        cx = random_state.randint(int(rad), img.shape[1] - int(rad))
        cy = random_state.randint(int(rad), img.shape[0] - int(rad))

        slices = np.linspace(0, 2 * np.pi, num_br + 1)
        angles = [slices[i] + random_state.rand() * (slices[i + 1] - slices[i])
                  for i in range(num_br)]
        tips = np.array([[int(cx + rad * np.cos(a)),
                          int(cy + rad * np.sin(a))] for a in angles])
        pts = np.vstack([[cx, cy], tips])

        col = get_random_color(background_color)
        thickness = random_state.randint(*thickness_rng)
        for t in tips:
            cv.line(img, (cx, cy), (t[0], t[1]), col, thickness)

        all_pts.append(pts)

    return np.vstack(all_pts) if all_pts else np.empty((0, 2), np.int32)


def draw_checkerboard(img: np.ndarray, min_rows: int = 3, min_cols: int = 3, max_rows: int = 7, max_cols: int = 7,
                      transform_params: Tuple[float, float] = (0.05, 0.15)) -> np.ndarray:
    """Draws a distorted checkerboard pattern on an image and returns its corner points.

    Generates a grid, warps it using affine and perspective transformations,
    fills alternating cells with contrasting colors, and draws random lines
    on the board boundaries.

    Args:
        img: The input image as a NumPy array (modified in place).
        max_rows: The maximum number of rows + 1 for the checkerboard.
        max_cols: The maximum number of columns + 1 for the checkerboard.
        transform_params: A tuple (min_ratio, max_ratio_add) defining the range of
                          transformation magnitudes relative to image size.

    Returns:
        A NumPy array of the corner points of the checkerboard cells that are inside the image (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        min_cols *= 2
        max_rows *= 2
        min_rows *= 2
        max_cols *= 2
    background_color: int = int(np.mean(img))

    # Create the base grid of points (x, y)
    rows: int = random_state.randint(min_rows, max_rows)  # Number of rows
    cols: int = random_state.randint(min_cols, max_cols)  # Number of columns
    s: int = min((img.shape[1] - 1) // cols,
                 (img.shape[0] - 1) // rows)  # Size of a cell
    x_coord: np.ndarray = np.tile(
        np.arange(cols + 1), rows + 1).reshape(((rows + 1) * (cols + 1), 1))
    y_coord: np.ndarray = np.repeat(
        np.arange(rows + 1), cols + 1).reshape(((rows + 1) * (cols + 1), 1))
    points: np.ndarray = s * np.concatenate([x_coord, y_coord], axis=1)

    # Warp the grid using an affine transformation and then a homography
    alpha_affine: float = np.max(
        img.shape) * (transform_params[0] + random_state.rand() * transform_params[1])
    center_square: np.ndarray = np.float32(img.shape) // 2
    min_dim: int = min(img.shape)
    square_size: int = min_dim // 3

    # Define source points for transformations
    pts1: np.ndarray = np.float32([center_square + square_size,
                                   [center_square[0] + square_size,
                                       center_square[1] - square_size],
                                   center_square - square_size,
                                   [center_square[0] - square_size, center_square[1] + square_size]])
    # Define destination points for affine transform
    pts2_affine: np.ndarray = pts1 + \
        random_state.uniform(-alpha_affine, alpha_affine,
                             size=pts1.shape).astype(np.float32)
    affine_transform: np.ndarray = cv.getAffineTransform(
        pts1[:3], pts2_affine[:3])

    # Define destination points for perspective transform
    pts2_perspective: np.ndarray = pts1 + \
        random_state.uniform(-alpha_affine / 2, alpha_affine /
                             2, size=pts1.shape).astype(np.float32)
    perspective_transform: np.ndarray = cv.getPerspectiveTransform(
        pts1, pts2_perspective)

    # Apply the affine transformation
    # Convert points to homogeneous coordinates for matrix multiplication
    points_homogeneous: np.ndarray = np.transpose(np.concatenate(
        (points, np.ones(((rows + 1) * (cols + 1), 1))), axis=1))
    warped_points_affine: np.ndarray = np.transpose(
        np.dot(affine_transform, points_homogeneous))

    # Apply the homography
    # Apply perspective transformation using matrix multiplication and normalization
    warped_col0: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[0, :2]), axis=1),
                                     perspective_transform[0, 2])
    warped_col1: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[1, :2]), axis=1),
                                     perspective_transform[1, 2])
    warped_col2: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[2, :2]), axis=1),
                                     perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points: np.ndarray = np.concatenate(
        [warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the checkerboard cells with alternating colors
    colors: np.ndarray = np.zeros((rows * cols,), np.int32)
    for i in range(rows):
        for j in range(cols):
            # Get a color that contrasts with the neighboring cells (left and top)
            if i == 0 and j == 0:
                col: int = get_random_color(background_color)
            else:
                neighboring_colors: List[int] = []
                if i != 0:
                    neighboring_colors.append(colors[(i-1) * cols + j])
                if j != 0:
                    neighboring_colors.append(colors[i * cols + j - 1])
                col = get_different_color(np.array(neighboring_colors))
            colors[i * cols + j] = col

            # Define the four corners of the current cell
            cell_corners: np.ndarray = np.array([
                (warped_points[i * (cols + 1) + j, 0],
                 warped_points[i * (cols + 1) + j, 1]),
                (warped_points[i * (cols + 1) + j + 1, 0],
                 warped_points[i * (cols + 1) + j + 1, 1]),
                (warped_points[(i + 1) * (cols + 1) + j + 1, 0],
                 warped_points[(i + 1) * (cols + 1) + j + 1, 1]),
                (warped_points[(i + 1) * (cols + 1) + j, 0],
                 warped_points[(i + 1) * (cols + 1) + j, 1])
            ])
            cv.fillConvexPoly(img, cell_corners, col)

    # Draw random lines on the boundaries of the board
    nb_rows_lines: int = random_state.randint(2, rows + 2)
    nb_cols_lines: int = random_state.randint(2, cols + 2)
    thickness: int = random_state.randint(
        int(min_dim * 0.01), int(min_dim * 0.015))

    for _ in range(nb_rows_lines):
        row_idx: int = random_state.randint(rows + 1)
        col_idx1: int = random_state.randint(cols + 1)
        col_idx2: int = random_state.randint(cols + 1)
        col: int = get_random_color(background_color)
        cv.line(img, (warped_points[row_idx * (cols + 1) + col_idx1, 0],
                      warped_points[row_idx * (cols + 1) + col_idx1, 1]),
                (warped_points[row_idx * (cols + 1) + col_idx2, 0],
                 warped_points[row_idx * (cols + 1) + col_idx2, 1]),
                col, thickness)
    for _ in range(nb_cols_lines):
        col_idx: int = random_state.randint(cols + 1)
        row_idx1: int = random_state.randint(rows + 1)
        row_idx2: int = random_state.randint(rows + 1)
        col: int = get_random_color(background_color)
        cv.line(img, (warped_points[row_idx1 * (cols + 1) + col_idx, 0],
                      warped_points[row_idx1 * (cols + 1) + col_idx, 1]),
                (warped_points[row_idx2 * (cols + 1) + col_idx, 0],
                 warped_points[row_idx2 * (cols + 1) + col_idx, 1]),
                col, thickness)

    # Keep only the points that fall inside the image boundaries after warping
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_stripes(img: np.ndarray, min_nb_cols: int = 5, max_nb_cols: int = 13, min_width_ratio: float = 0.04,
                 transform_params: Tuple[float, float] = (0.05, 0.15)) -> np.ndarray:
    """Draws a series of distorted stripes on an image and returns their corner points.

    Generates vertical stripes within a larger rectangle, warps the rectangle
    and stripes using affine and perspective transformations, fills them with
    contrasting colors, and draws random lines on stripe boundaries.

    Args:
        img: The input image as a NumPy array (modified in place).
        max_nb_cols: The maximum number of stripes to be drawn.
        min_width_ratio: The minimal width of a stripe as a ratio of the smallest image dimension.
        transform_params: A tuple (min_ratio, max_ratio_add) defining the range of
                          transformation magnitudes relative to image size.

    Returns:
        A NumPy array of the corner points of the stripes that are inside the image (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    if cfg.hard:
        min_nb_cols *= 2
        max_nb_cols *= 2
        min_width_ratio /= 2
    background_color: int = int(np.mean(img))

    board_size: Tuple[int, int] = (int(img.shape[0] * (1 + random_state.rand())),
                                   int(img.shape[1] * (1 + random_state.rand())))
    col_count: int = random_state.randint(min_nb_cols, max_nb_cols)
    cols_x_coords: np.ndarray = np.concatenate([board_size[1] * random_state.rand(col_count - 1),
                                                np.array([0, board_size[1] - 1])], axis=0)
    cols_x_coords = np.unique(cols_x_coords.astype(int))

    min_dim: int = min(img.shape)
    min_width: float = min_dim * min_width_ratio
    cols_x_coords = cols_x_coords[(np.concatenate([cols_x_coords[1:],
                                                   np.array([board_size[1] + min_width])],
                                                  axis=0) - cols_x_coords) >= min_width]
    col_count = cols_x_coords.shape[0] - 1

    cols_x_coords = np.reshape(cols_x_coords, (col_count + 1, 1))
    points1: np.ndarray = np.concatenate(
        [cols_x_coords, np.zeros((col_count + 1, 1), np.int32)], axis=1)
    points2: np.ndarray = np.concatenate([cols_x_coords,
                                          (board_size[0] - 1) * np.ones((col_count + 1, 1), np.int32)],
                                         axis=1)
    points: np.ndarray = np.concatenate([points1, points2], axis=0)

    alpha_affine: float = np.max(
        img.shape) * (transform_params[0] + random_state.rand() * transform_params[1])
    center_square: np.ndarray = np.float32(img.shape) // 2
    square_size: int = min(img.shape) // 3
    pts1: np.ndarray = np.float32([center_square + square_size,
                                   [center_square[0] + square_size,
                                       center_square[1] - square_size],
                                   center_square - square_size,
                                   [center_square[0] - square_size, center_square[1] + square_size]])
    pts2_affine: np.ndarray = pts1 + \
        random_state.uniform(-alpha_affine, alpha_affine,
                             size=pts1.shape).astype(np.float32)
    affine_transform: np.ndarray = cv.getAffineTransform(
        pts1[:3], pts2_affine[:3])
    pts2_perspective: np.ndarray = pts1 + \
        random_state.uniform(-alpha_affine / 2, alpha_affine /
                             2, size=pts1.shape).astype(np.float32)
    perspective_transform: np.ndarray = cv.getPerspectiveTransform(
        pts1, pts2_perspective)

    # Apply the affine transformation
    points_homogeneous: np.ndarray = np.transpose(np.concatenate(
        (points, np.ones((2 * (col_count + 1), 1))), axis=1))
    warped_points_affine: np.ndarray = np.transpose(
        np.dot(affine_transform, points_homogeneous))

    # Apply the homography
    warped_col0: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[0, :2]), axis=1),
                                     perspective_transform[0, 2])
    warped_col1: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[1, :2]), axis=1),
                                     perspective_transform[1, 2])
    warped_col2: np.ndarray = np.add(np.sum(np.multiply(warped_points_affine, perspective_transform[2, :2]), axis=1),
                                     perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points: np.ndarray = np.concatenate(
        [warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangular stripes
    color: int = get_random_color(background_color)
    for i in range(col_count):
        # Shift color for next stripe to ensure variation
        color = (color + 128 + random_state.randint(-30, 30)) % 256
        # Define the four corners of the current stripe rectangle
        stripe_corners: np.ndarray = np.array([
            (warped_points[i, 0], warped_points[i, 1]),
            (warped_points[i+1, 0], warped_points[i+1, 1]),
            (warped_points[i+col_count+2, 0], warped_points[i+col_count+2, 1]),
            (warped_points[i+col_count+1, 0], warped_points[i+col_count+1, 1])
        ])
        cv.fillConvexPoly(img, stripe_corners, color)

    # Draw random lines on the boundaries of the stripes
    nb_rows_lines: int = random_state.randint(2, 5)
    nb_cols_lines: int = random_state.randint(2, col_count + 2)
    thickness: int = random_state.randint(
        int(min_dim * 0.01), int(min_dim * 0.015))

    for _ in range(nb_rows_lines):
        row_idx: int = random_state.choice(
            [0, col_count + 1])  # Choose top or bottom edge
        col_idx1: int = random_state.randint(col_count + 1)
        col_idx2: int = random_state.randint(col_count + 1)
        color = get_random_color(background_color)
        cv.line(img, (warped_points[row_idx + col_idx1, 0],
                      warped_points[row_idx + col_idx1, 1]),
                (warped_points[row_idx + col_idx2, 0],
                 warped_points[row_idx + col_idx2, 1]),
                color, thickness)
    for _ in range(nb_cols_lines):
        col_idx: int = random_state.randint(col_count + 1)
        color = get_random_color(background_color)
        cv.line(img, (warped_points[col_idx, 0],
                      warped_points[col_idx, 1]),
                (warped_points[col_idx + col_count + 1, 0],
                 warped_points[col_idx + col_count + 1, 1]),
                color, thickness)

    # Keep only the points that fall inside the image boundaries after warping
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_cube(img: np.ndarray, min_size_ratio: float = 0.2, min_angle_rot: float = numpy.pi / 10,
              scale_interval: Tuple[float, float] = (0.4, 0.6),
              trans_interval: Tuple[float, float] = (0.5, 0.2)) -> np.ndarray:
    """Draws a 2D projection of a 3D cube on an image and returns its visible corner points.

    Generates a 3D cube, applies random rotations, scaling, and translation,
    then projects it onto a 2D plane. The visible faces are filled, and edges are drawn.

    Args:
        img: The input image as a NumPy array (modified in place).
        min_size_ratio: The minimum side length of the cube as a ratio of the smallest image dimension.
        min_angle_rot: The minimum angle of rotation (in radians) applied along each axis.
        scale_interval: A tuple (min_scale, scale_range) for random scaling factors.
        trans_interval: A tuple (base_ratio, range_ratio) for random translation,
                        relative to image dimensions.

    Returns:
        A NumPy array of the visible corner points of the drawn cube that are inside the image (N, 2).
    """
    if random_state is None:
        raise RuntimeError(
            "random_state not set. Call set_random_state first.")
    background_color: int = int(np.mean(img))
    min_dim: int = min(img.shape[:2])
    min_side: float = min_dim * min_size_ratio

    lx: float = min_side + random_state.rand() * 2 * min_dim / 3
    ly: float = min_side + random_state.rand() * 2 * min_dim / 3
    lz: float = min_side + random_state.rand() * 2 * min_dim / 3

    # Define cube vertices (origin at (0,0,0))
    cube: np.ndarray = np.array([[0, 0, 0],
                                 [lx, 0, 0],
                                 [0, ly, 0],
                                 [lx, ly, 0],
                                 [0, 0, lz],
                                 [lx, 0, lz],
                                 [0, ly, lz],
                                 [lx, ly, lz]])

    # Generate random rotation angles for each axis
    rot_angles: np.ndarray = random_state.rand(
        3) * (3 * numpy.pi / 10.) + min_angle_rot

    # Define rotation matrices
    rotation_1: np.ndarray = np.array([[numpy.cos(rot_angles[0]), -numpy.sin(rot_angles[0]), 0],
                                       [numpy.sin(rot_angles[0]), numpy.cos(
                                           rot_angles[0]), 0],
                                       [0, 0, 1]])  # Rotation around Z-axis
    rotation_2: np.ndarray = np.array([[1, 0, 0],
                                       [0, numpy.cos(rot_angles[1]), -
                                        numpy.sin(rot_angles[1])],
                                       # Rotation around X-axis
                                       [0, numpy.sin(rot_angles[1]), numpy.cos(rot_angles[1])]])
    rotation_3: np.ndarray = np.array([[numpy.cos(rot_angles[2]), 0, -numpy.sin(rot_angles[2])],
                                       [0, 1, 0],
                                       # Rotation around Y-axis
                                       [numpy.sin(rot_angles[2]), 0, numpy.cos(rot_angles[2])]])

    # Define scaling matrix
    scaling: np.ndarray = np.array([[scale_interval[0] + random_state.rand() * scale_interval[1], 0, 0],
                                    [0, scale_interval[0] + random_state.rand()
                                     * scale_interval[1], 0],
                                    [0, 0, scale_interval[0] + random_state.rand() * scale_interval[1]]])

    # Define translation vector
    trans: np.ndarray = np.array([img.shape[1] * trans_interval[0] +
                                  random_state.randint(int(-img.shape[1] * trans_interval[1]),
                                                       int(img.shape[1] * trans_interval[1])),
                                  img.shape[0] * trans_interval[0] +
                                  random_state.randint(int(-img.shape[0] * trans_interval[1]),
                                                       int(img.shape[0] * trans_interval[1])),
                                  0])  # Z-translation is 0 for 2D projection

    # Apply transformations: rotation, scaling, and translation (order matters)
    # The order of matrix multiplications (rotation_3 * rotation_2 * rotation_1) is applied first, then scaling.
    # Finally, translation is added to the transformed vertices.
    transformed_cube: np.ndarray = trans + np.transpose(np.dot(scaling,
                                                               np.dot(rotation_1,
                                                                      np.dot(rotation_2,
                                                                             np.dot(rotation_3,
                                                                                    np.transpose(cube))))))

    # Project on the plane z=0 by taking only the first two dimensions
    cube_2d: np.ndarray = transformed_cube[:, :2].astype(int)

    # In a standard cube vertex ordering, vertex 0 is typically the "hidden" one if oriented towards +ve axes
    # and vertex 7 is the "front" one. The visible corners are typically 1,2,3,4,5,6,7 assuming 0 is hidden.
    # Get rid of the hidden corner (index 0)
    points: np.ndarray = cube_2d[1:, :]

    # Define the three visible faces based on standard cube vertex indexing
    faces: np.ndarray = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

    # Fill the faces and draw the contours
    col_face: int = get_random_color(background_color)
    for i in [0, 1, 2]:  # Iterate through the three visible faces
        cv.fillPoly(img, [cube_2d[faces[i]].reshape((-1, 1, 2))], col_face)

    # Draw edges with a contrasting color
    thickness: int = random_state.randint(
        int(min_dim * 0.003), int(min_dim * 0.015))
    for i in [0, 1, 2]:  # Iterate through the faces
        for j in [0, 1, 2, 3]:  # Iterate through edges of each face
            # Color that contrasts with the face color
            col_edge: int = (col_face + 128 +
                             random_state.randint(-64, 64)) % 256
            cv.line(img, (cube_2d[faces[i][j], 0], cube_2d[faces[i][j], 1]),
                    (cube_2d[faces[i][(j + 1) % 4], 0],
                     cube_2d[faces[i][(j + 1) % 4], 1]),
                    col_edge, thickness)

    # Keep only the points that fall inside the image boundaries
    points = keep_points_inside(points, img.shape[:2])
    return points


def draw_random_cubes(img: np.ndarray, n_min: int = 2, n_max: int = 4) -> np.ndarray:
    """Draws multiple random cubes on an image, ensuring a minimum number are successfully drawn.

    Attempts to draw cubes using the `draw_cube` function and collects the corner points.
    Retries drawing until at least `n_min` cubes have visible corners or max attempts are reached.

    Args:
        img: The input image as a NumPy array (modified in place).
        n_min: The minimum number of cubes desired.
        n_max: The maximum number of attempts to draw a cube.

    Returns:
        A NumPy array of all visible corner points from the drawn cubes (N, 2).
    """
    if cfg.hard:
        n_min *= 3
        n_max *= 3
    corners_all: List[np.ndarray] = []
    attempts: int = 0
    # Max attempts to avoid infinite loops
    while len(corners_all) < n_min and attempts < n_max * 3:
        # Draw a cube with varied parameters
        pts: np.ndarray = draw_cube(img,
                                    min_size_ratio=0.18,
                                    trans_interval=(0.3, 0.4))
        if pts.shape[0] > 0:  # Check if any points were successfully drawn
            corners_all.append(pts)
        attempts += 1
    # Concatenate all collected points or return an empty array if none were drawn
    return np.vstack(corners_all) if corners_all else np.empty((0, 2), np.int32)


def draw_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Applies uniform random noise (0-255) to an image.

    Args:
        img: The input image as a NumPy array (modified in place).

    Returns:
        An empty NumPy array of shape (0, 2) and dtype int32 (no specific points are tracked).
    """
    cv.randu(img, 0, 255)
    return np.empty((0, 2), dtype=np.int32)


def draw_interest_points(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Converts a grayscale image to RGB and draws specified interest points in green.

    Args:
        img: The input grayscale image as a NumPy array.
        points: A NumPy array of points (N, 2) to be drawn.

    Returns:
        The image converted to RGB with the interest points drawn as green circles.
    """
    # Convert grayscale image to 3-channel RGB
    img_rgb: np.ndarray = np.stack([img, img, img], axis=2)
    # Draw green circles at each interest point
    for i in range(points.shape[0]):
        # (0,255,0) is green, -1 for filled
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb
