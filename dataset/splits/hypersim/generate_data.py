import os

def generate_file_list(base_path, folders, train_file, test_file):
    with open(train_file, 'w') as train, open(test_file, 'w') as test:
        for folder in folders:
            folder_path = os.path.join(base_path, folder, "images")
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if "tonemap" in file:
                        tonemap_file = os.path.join(root, file)
                        depth_file = tonemap_file.replace("scene_cam_00_final_preview", "scene_cam_00_geometry_preview").replace("tonemap.jpg", "semantic.png")
                        if os.path.exists(depth_file):
                            line = f"{tonemap_file} {depth_file}\n"
                            train.write(line)
                            test.write(line)

# Configuration
base_path = "/mnt/all"
folders = [f"ai_001_00{i}" for i in range(1, 6)]
train_file = "train.txt"
test_file = "val.txt"

# Generate file lists
generate_file_list(base_path, folders, train_file, test_file)

