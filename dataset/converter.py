def convert_polygon_to_yolo_bbox(polygon_vertices, class_index):
    # Calculate bounding box coordinates
    min_x, min_y = min(polygon_vertices, key=lambda vertex: vertex[0])[0], min(polygon_vertices, key=lambda vertex: vertex[1])[1]
    max_x, max_y = max(polygon_vertices, key=lambda vertex: vertex[0])[0], max(polygon_vertices, key=lambda vertex: vertex[1])[1]

    # Calculate bounding box parameters
    x_center = (min_x + max_x) / 2.0
    y_center = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y

    # Return the YOLO bbox annotation string
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Example usage:
class_index = 0
polygon_vertices=[]
yolo_bbox_annotation = convert_polygon_to_yolo_bbox(polygon_vertices, class_index)
print(yolo_bbox_annotation)
