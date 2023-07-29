def read_text_file_to_numpy(file_path):
    data = []

    # Step 1: Read the text file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Step 2: Split each line into individual elements
            elements = line.strip().split()
            # Step 3: Convert the elements into numerical values (if needed)
            numerical_elements = [float(element) for element in elements]  # Convert to float, change data type if needed

            className,polygon_vertices = numerical_elements[0],numerical_elements[1:]
            min_x, min_y = min(polygon_vertices[::2]), min(polygon_vertices[1::2])
            max_x, max_y = max(polygon_vertices[::2]), max(polygon_vertices[1::2])

            # Calculate bounding box parameters
            x_center = (min_x + max_x) / 2.0
            y_center = (min_y + max_y) / 2.0
            width = max_x - min_x
            height = max_y - min_y

            # Return the YOLO bbox annotation string
            print(f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            data.append([x_center,y_center,width,height,className])
    return data


# Usage example:
file_path = '0.txt'  # Replace 'data.txt' with the path to your text file
numpy_array = read_text_file_to_numpy(file_path)
print(numpy_array)
