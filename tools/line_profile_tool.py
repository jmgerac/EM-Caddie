import numpy as np
import cv2

def get_line_profile(img, p1, p2):
    """
    Extract intensity profile along a line between two points.
    
    :param img: Input image (grayscale or BGR)
    :param p1: Start point (x, y)
    :param p2: End point (x, y)
    :return: distances (array), intensities (array)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Get line length in pixels (use float for precision)
    length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
    if length == 0:
        return np.array([0]), np.array([gray[int(p1[1]), int(p1[0])]])
    
    # Sample at least one point per pixel, but ensure we cover the entire range
    # Use ceil to ensure we get at least length+1 points to cover endpoints
    num_samples = max(int(np.ceil(length)) + 1, 2)
    
    # Create array of x and y coordinates along the line
    # Use endpoint=True to ensure we include both p1 and p2
    x = np.linspace(p1[0], p2[0], num_samples, endpoint=True)
    y = np.linspace(p1[1], p2[1], num_samples, endpoint=True)
    
    # Extract pixel values along the line
    intensities = []
    distances = []
    cumulative_dist = 0.0
    
    for i in range(num_samples):
        xi, yi = int(round(x[i])), int(round(y[i]))
        
        # Calculate distance from start point
        if i > 0:
            segment_dist = np.hypot(x[i] - x[i-1], y[i] - y[i-1])
            cumulative_dist += segment_dist
        
        distances.append(cumulative_dist)
        
        if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
            intensities.append(gray[yi, xi])
        else:
            intensities.append(0)
    
    return np.array(distances), np.array(intensities)


def get_polyline_profile(img, points):
    """
    Extract intensity profile along a polyline (list of x, y points).
    Consecutive points are joined with straight segments.
    For 2-point lines, ensures complete coverage without gaps.

    :param img: Input image (grayscale or BGR)
    :param points: Iterable of (x, y) points; at least two are required
    :return: distances (array), intensities (array), sample_coords (array of (x, y))
    """
    if len(points) < 2:
        return np.array([]), np.array([]), np.empty((0, 2))

    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    distances: list[float] = []
    intensities: list[float] = []
    sample_coords: list[tuple[int, int]] = []
    running_dist = 0.0

    for idx in range(len(points) - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]

        segment_len = float(np.hypot(x2 - x1, y2 - y1))
        
        # For accurate sampling, use at least one sample per pixel, but ensure we cover the full segment
        # Use endpoint=True to ensure we include the endpoint
        num_samples = max(int(np.ceil(segment_len)) + 1, 2)
        
        xs = np.linspace(x1, x2, num_samples, endpoint=True)
        ys = np.linspace(y1, y2, num_samples, endpoint=True)

        for step_idx in range(num_samples):
            # Skip the first point of subsequent segments to avoid duplicate sampling
            if idx > 0 and step_idx == 0:
                continue

            xi, yi = int(round(xs[step_idx])), int(round(ys[step_idx]))
            
            # Calculate distance from start of current segment
            if step_idx == 0 and idx == 0:
                # First point of first segment - distance is 0
                segment_dist = 0.0
            elif step_idx == 0:
                # First point of subsequent segment - distance from previous segment's endpoint
                prev_x, prev_y = points[idx]
                segment_dist = np.hypot(xs[step_idx] - prev_x, ys[step_idx] - prev_y)
            else:
                # Distance from previous sampled point
                prev_x = xs[step_idx - 1]
                prev_y = ys[step_idx - 1]
                segment_dist = np.hypot(xs[step_idx] - prev_x, ys[step_idx] - prev_y)
            
            running_dist += segment_dist

            if 0 <= xi < gray.shape[1] and 0 <= yi < gray.shape[0]:
                intensities.append(gray[yi, xi])
            else:
                intensities.append(0)

            distances.append(running_dist)
            sample_coords.append((xi, yi))

    return np.array(distances), np.array(intensities), np.array(sample_coords)


def get_full_image_line(img, angle_degrees, vertical_offset=0.5):
    """
    Get a line that spans the entire image at a given angle.
    Angle is relative to the bottom of the image (0 degrees = horizontal, 90 = vertical).
    
    :param img: Input image
    :param angle_degrees: Angle in degrees (0 = horizontal, 90 = vertical from bottom)
    :param vertical_offset: Vertical position offset (0.0 = top, 0.5 = center, 1.0 = bottom)
    :return: p1, p2 (line endpoints)
    """
    if len(img.shape) == 3:
        height, width = img.shape[:2]
    else:
        height, width = img.shape
    
    # Convert angle: 0 = horizontal (left-right), 90 = vertical (bottom-top)
    # In image coordinates: 0 degrees = horizontal, positive = counterclockwise
    angle_rad = np.radians(angle_degrees)
    
    # Calculate line that spans entire image
    # Use vertical_offset to position the line vertically (0 = top, 1 = bottom)
    # The line will pass through the point (width/2, height * vertical_offset)
    center_y = height * vertical_offset
    center_x = width / 2
    
    # Calculate line direction vector (normalized)
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)  # Negative because y increases downward
    
    # Find intersections with image boundaries by extending the line from the center point
    # Line equation: (x, y) = (center_x, center_y) + t * (dx, dy)
    intersections = []
    
    # Check intersection with top edge (y = 0)
    if abs(dy) > 1e-6:
        t = (0 - center_y) / dy
        x = center_x + t * dx
        if 0 <= x <= width:
            intersections.append((x, 0))
    
    # Check intersection with bottom edge (y = height)
    if abs(dy) > 1e-6:
        t = (height - center_y) / dy
        x = center_x + t * dx
        if 0 <= x <= width:
            intersections.append((x, height))
    
    # Check intersection with left edge (x = 0)
    if abs(dx) > 1e-6:
        t = (0 - center_x) / dx
        y = center_y + t * dy
        if 0 <= y <= height:
            intersections.append((0, y))
    
    # Check intersection with right edge (x = width)
    if abs(dx) > 1e-6:
        t = (width - center_x) / dx
        y = center_y + t * dy
        if 0 <= y <= height:
            intersections.append((width, y))
    
    # Remove duplicates and ensure we have exactly 2 points
    unique_intersections = []
    for point in intersections:
        if point not in unique_intersections:
            unique_intersections.append(point)
    
    if len(unique_intersections) >= 2:
        # Use the two points that are farthest apart (should be the endpoints)
        # For a line spanning the image, these should be on opposite edges
        p1, p2 = unique_intersections[0], unique_intersections[1]
        
        # If we have more than 2, find the two that are farthest apart
        if len(unique_intersections) > 2:
            max_dist = 0
            for i in range(len(unique_intersections)):
                for j in range(i + 1, len(unique_intersections)):
                    dist = np.hypot(unique_intersections[i][0] - unique_intersections[j][0],
                                  unique_intersections[i][1] - unique_intersections[j][1])
                    if dist > max_dist:
                        max_dist = dist
                        p1, p2 = unique_intersections[i], unique_intersections[j]
        
        # Ensure p1 is the one with smaller y (closer to top) or smaller x if same y
        if p1[1] > p2[1] or (abs(p1[1] - p2[1]) < 1e-6 and p1[0] > p2[0]):
            p1, p2 = p2, p1
        return p1, p2
    elif len(unique_intersections) == 1:
        # Only one intersection (line goes through a corner)
        # Extend in both directions
        corner = unique_intersections[0]
        max_dist = np.hypot(width, height)
        # Extend backwards and forwards
        p1 = (corner[0] - max_dist * dx, corner[1] - max_dist * dy)
        p2 = (corner[0] + max_dist * dx, corner[1] + max_dist * dy)
        # Clamp to image bounds
        p1 = (max(0, min(width, p1[0])), max(0, min(height, p1[1])))
        p2 = (max(0, min(width, p2[0])), max(0, min(height, p2[1])))
        return p1, p2
    else:
        # Fallback: horizontal line at the vertical offset position
        return (0, int(center_y)), (width, int(center_y))


def get_shape_perimeter_points(shape_type, center, size, angle_degrees=0, start_fraction=0.0, end_fraction=1.0):
    """
    Get points along the perimeter of a shape, starting from bottom point, going right around.
    
    :param shape_type: 'circle', 'triangle', 'square', 'pentagon', 'hexagon', 'heptagon', or 'octagon'
    :param center: (x, y) center of shape
    :param size: radius for circle, side length for polygons
    :param angle_degrees: rotation angle relative to bottom of image
    :param start_fraction: Start position along perimeter (0.0 to 1.0)
    :param end_fraction: End position along perimeter (0.0 to 1.0)
    :return: array of (x, y) points along perimeter
    """
    cx, cy = center
    angle_rad = np.radians(angle_degrees)
    
    if shape_type == 'circle':
        # Generate points around circle
        num_points = int(2 * np.pi * size)  # More points for larger circles
        num_points = max(32, min(num_points, 360))  # Between 32 and 360 points
        
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        # Start from bottom (270 degrees in standard coordinates, but y increases downward)
        # In image coordinates, bottom is at angle -pi/2 (or 3*pi/2)
        start_angle = -np.pi / 2
        angles = angles + start_angle + angle_rad
        
        points = []
        for angle in angles:
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.append((x, y))
        
        # Apply range selection
        total_points = len(points)
        start_idx = int(start_fraction * total_points)
        end_idx = int(end_fraction * total_points)
        
        if start_fraction < end_fraction:
            return np.array(points[start_idx:end_idx])
        else:
            # Handle wrap-around case
            return np.array(points[start_idx:] + points[:end_idx])
    
    elif shape_type == 'square':
        # Square vertices (before rotation)
        half_size = size / 2
        vertices = [
            (cx - half_size, cy - half_size),  # Top-left
            (cx + half_size, cy - half_size),  # Top-right
            (cx + half_size, cy + half_size),  # Bottom-right
            (cx - half_size, cy + half_size),  # Bottom-left
        ]
        
        # Rotate vertices
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotated_vertices = []
        for vx, vy in vertices:
            # Translate to origin, rotate, translate back
            vx_rel = vx - cx
            vy_rel = vy - cy
            vx_rot = vx_rel * cos_a - vy_rel * sin_a
            vy_rot = vx_rel * sin_a + vy_rel * cos_a
            rotated_vertices.append((vx_rot + cx, vy_rot + cy))
        
        # Find bottom point (highest y value)
        bottom_idx = max(range(len(rotated_vertices)), key=lambda i: rotated_vertices[i][1])
        
        # Reorder starting from bottom, going right (clockwise)
        reordered = rotated_vertices[bottom_idx:] + rotated_vertices[:bottom_idx]
        
        # Generate points along edges
        points = []
        for i in range(len(reordered)):
            p1 = reordered[i]
            p2 = reordered[(i + 1) % len(reordered)]
            # Interpolate along edge
            num_edge_points = max(10, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            for j in range(num_edge_points):
                t = j / num_edge_points
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                points.append((x, y))
        
        # Apply range selection
        total_points = len(points)
        start_idx = int(start_fraction * total_points)
        end_idx = int(end_fraction * total_points)
        
        if start_fraction < end_fraction:
            return np.array(points[start_idx:end_idx])
        else:
            # Handle wrap-around case
            return np.array(points[start_idx:] + points[:end_idx])
    
    elif shape_type == 'triangle':
        # Equilateral triangle vertices (before rotation)
        # Height of equilateral triangle: h = s * sqrt(3) / 2
        height = size * np.sqrt(3) / 2
        half_size = size / 2
        
        # Triangle pointing up initially
        vertices = [
            (cx, cy - 2 * height / 3),  # Top vertex
            (cx - half_size, cy + height / 3),  # Bottom-left
            (cx + half_size, cy + height / 3),  # Bottom-right
        ]
        
        # Rotate vertices
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotated_vertices = []
        for vx, vy in vertices:
            vx_rel = vx - cx
            vy_rel = vy - cy
            vx_rot = vx_rel * cos_a - vy_rel * sin_a
            vy_rot = vx_rel * sin_a + vy_rel * cos_a
            rotated_vertices.append((vx_rot + cx, vy_rot + cy))
        
        # Find bottom point (highest y value)
        bottom_idx = max(range(len(rotated_vertices)), key=lambda i: rotated_vertices[i][1])
        
        # Reorder starting from bottom, going right (clockwise)
        reordered = rotated_vertices[bottom_idx:] + rotated_vertices[:bottom_idx]
        
        # Generate points along edges
        points = []
        for i in range(len(reordered)):
            p1 = reordered[i]
            p2 = reordered[(i + 1) % len(reordered)]
            num_edge_points = max(10, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            for j in range(num_edge_points):
                t = j / num_edge_points
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                points.append((x, y))
        
        # Apply range selection
        total_points = len(points)
        start_idx = int(start_fraction * total_points)
        end_idx = int(end_fraction * total_points)
        
        if start_fraction < end_fraction:
            return np.array(points[start_idx:end_idx])
        else:
            # Handle wrap-around case
            return np.array(points[start_idx:] + points[:end_idx])
    
    elif shape_type in ['pentagon', 'hexagon', 'heptagon', 'octagon']:
        # Regular polygon vertices
        num_sides = {'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8}[shape_type]
        
        # Calculate radius from center to vertex for regular polygon
        # For a regular n-gon with side length s, radius r = s / (2 * sin(π/n))
        radius = size / (2 * np.sin(np.pi / num_sides))
        
        # Generate vertices
        vertices = []
        for i in range(num_sides):
            # Start from top (angle -π/2), then rotate
            angle_vertex = -np.pi / 2 + (2 * np.pi * i / num_sides) + angle_rad
            x = cx + radius * np.cos(angle_vertex)
            y = cy + radius * np.sin(angle_vertex)
            vertices.append((x, y))
        
        # Find bottom point (highest y value)
        bottom_idx = max(range(len(vertices)), key=lambda i: vertices[i][1])
        
        # Reorder starting from bottom, going right (clockwise)
        reordered = vertices[bottom_idx:] + vertices[:bottom_idx]
        
        # Generate points along edges
        all_points = []
        for i in range(len(reordered)):
            p1 = reordered[i]
            p2 = reordered[(i + 1) % len(reordered)]
            num_edge_points = max(10, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
            for j in range(num_edge_points):
                t = j / num_edge_points
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                all_points.append((x, y))
        
        # Apply range selection
        total_points = len(all_points)
        start_idx = int(start_fraction * total_points)
        end_idx = int(end_fraction * total_points)
        
        if start_fraction < end_fraction:
            return np.array(all_points[start_idx:end_idx])
        else:
            # Handle wrap-around case
            return np.array(all_points[start_idx:] + all_points[:end_idx])
    
    return np.array([])


def get_shape_perimeter_profile(img, shape_type, center, size, angle_degrees=0, start_fraction=0.0, end_fraction=1.0):
    """
    Get intensity profile along shape perimeter.
    
    :param img: Input image
    :param shape_type: 'circle', 'triangle', 'square', 'pentagon', 'hexagon', 'heptagon', or 'octagon'
    :param center: (x, y) center of shape
    :param size: radius for circle, side length for polygons
    :param angle_degrees: rotation angle
    :param start_fraction: Start position along perimeter (0.0 to 1.0)
    :param end_fraction: End position along perimeter (0.0 to 1.0)
    :return: distances (array), intensities (array)
    """
    # Get perimeter points
    points = get_shape_perimeter_points(shape_type, center, size, angle_degrees, start_fraction, end_fraction)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Extract intensities along perimeter
    intensities = []
    cumulative_distances = [0]
    total_dist = 0
    
    for i in range(len(points)):
        x, y = int(points[i][0]), int(points[i][1])
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            intensities.append(gray[y, x])
        else:
            intensities.append(0)
        
        if i > 0:
            dist = np.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
            total_dist += dist
            cumulative_distances.append(total_dist)
    
    return np.array(cumulative_distances), np.array(intensities)


def draw_line_on_image(img, p1, p2, color=(0, 255, 0), thickness=2):
    """
    Draw a line on an image.
    
    :param img: Input image (will be copied)
    :param p1: Start point (x, y)
    :param p2: End point (x, y)
    :param color: Line color (BGR tuple)
    :param thickness: Line thickness
    :return: Image with line drawn
    """
    result = img.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    cv2.line(result, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    return result


def draw_shape_on_image(img, shape_type, center, size, angle_degrees=0, color=(0, 255, 0), thickness=2):
    """
    Draw a shape on an image.
    
    :param img: Input image (will be copied)
    :param shape_type: 'circle', 'triangle', 'square', 'pentagon', 'hexagon', 'heptagon', or 'octagon'
    :param center: (x, y) center of shape
    :param size: radius for circle, side length for polygons
    :param angle_degrees: rotation angle
    :param color: Shape color (BGR tuple)
    :param thickness: Line thickness
    :return: Image with shape drawn
    """
    result = img.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    cx, cy = int(center[0]), int(center[1])
    angle_rad = np.radians(angle_degrees)
    
    if shape_type == 'circle':
        cv2.circle(result, (cx, cy), int(size), color, thickness)
    
    elif shape_type == 'square':
        half_size = size / 2
        # Create square vertices
        vertices = np.array([
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size]
        ], dtype=np.float32)
        
        # Rotate
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        rotated_vertices = vertices @ rotation_matrix.T
        
        # Translate to center
        rotated_vertices[:, 0] += cx
        rotated_vertices[:, 1] += cy
        
        # Draw
        pts = rotated_vertices.astype(np.int32)
        cv2.polylines(result, [pts], True, color, thickness)
    
    elif shape_type == 'triangle':
        height = size * np.sqrt(3) / 2
        half_size = size / 2
        
        vertices = np.array([
            [0, -2 * height / 3],
            [-half_size, height / 3],
            [half_size, height / 3]
        ], dtype=np.float32)
        
        # Rotate
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        rotated_vertices = vertices @ rotation_matrix.T
        
        # Translate to center
        rotated_vertices[:, 0] += cx
        rotated_vertices[:, 1] += cy
        
        # Draw
        pts = rotated_vertices.astype(np.int32)
        cv2.polylines(result, [pts], True, color, thickness)
    
    elif shape_type in ['pentagon', 'hexagon', 'heptagon', 'octagon']:
        num_sides = {'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8}[shape_type]
        radius = size / (2 * np.sin(np.pi / num_sides))
        
        # Generate vertices
        vertices = []
        for i in range(num_sides):
            angle_vertex = -np.pi / 2 + (2 * np.pi * i / num_sides) + angle_rad
            x = cx + radius * np.cos(angle_vertex)
            y = cy + radius * np.sin(angle_vertex)
            vertices.append([x, y])
        
        # Draw
        pts = np.array(vertices, dtype=np.int32)
        cv2.polylines(result, [pts], True, color, thickness)
    
    return result

