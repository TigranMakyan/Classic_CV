def get_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) metric for two bounding boxes.

    Args:
        box1 (list or tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
        box2 (list or tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns:
        float: Intersection over Union (IoU) value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    intersection_width = min(x1_max, x2_max) - max(x1_min, x2_min)
    intersection_height = min(y1_max, y2_max) - max(y1_min, y2_min)
    intersection_area = max(0, intersection_width) * max(0, intersection_height)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou
