import os

def reformat_image_list(list_file_path):
    """
    Reformats the list_of_images.txt file to change from format:
    '2008_000002.jpg    tvmonitor'
    to:
    'tvmonitor/2008_000002.jpg    tvmonitor'
    
    Args:
        list_file_path: Path to the list_of_images.txt file
    """
    if not os.path.exists(list_file_path):
        print(f"Error: {list_file_path} not found")
        return
    
    # Read the current file
    with open(list_file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line and reformat
    reformatted_lines = []
    for line in lines:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_name, class_name = parts
                # Create new format with class name prepended to the image path
                reformatted_line = f"{class_name}/{image_name}\t{class_name}\n"
                reformatted_lines.append(reformatted_line)
            else:
                # Keep unchanged if format doesn't match
                reformatted_lines.append(line)
        else:
            # Keep empty lines
            reformatted_lines.append(line)
    
    # Write the reformatted content back to the file
    with open(list_file_path, 'w') as f:
        f.writelines(reformatted_lines)
    
    print(f"Successfully reformatted {list_file_path}")

if __name__ == "__main__":
    # Path to the VOC_val list file
    voc_val_list_file = r"c:\Users\jidel\U\AI\T2\T2images\VOC_val\list_of_images.txt"
    
    # Process the file
    reformat_image_list(voc_val_list_file)