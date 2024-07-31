import os

def generate_file_list(train_file, test_file):
    with open(train_file, 'w') as train, open(test_file, 'w') as test:
        rgb_folder = "/home/nicolas/repos/Depth-Anything-V2/metric_depth/data/RGB"
        seg_folder = "/home/nicolas/repos/Depth-Anything-V2/metric_depth/data/Seg"
        print(rgb_folder)
        for root, _, files in os.walk(rgb_folder):
            #print(files)
            for file in files:
                rgb_file = os.path.join(root, file)
                seg_file = os.path.join(seg_folder, os.path.relpath(rgb_file, rgb_folder)).replace('.jpg', '.jpg')
                
                if os.path.exists(seg_file):
                    line = f"{rgb_file} {seg_file}\n"
                    train.write(line)
                    test.write(line)

# Configuration
base_path = ""
train_file = "train.txt"
test_file = "val.txt"

# Generate file lists
generate_file_list(train_file, test_file)

