import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

color_to_gray_hypersim = {
    (174, 199, 232): 0,
    (152, 223, 138): 1,
    (31, 119, 180): 2,
    (255, 187, 120): 3,
    (188, 189, 34): 4,
    (140, 86, 75): 5,
    (255, 152, 150): 6,
    (214, 39, 40): 7,
    (197, 176, 213): 8,
    (148, 103, 189): 9,
    (196, 156, 148): 10,
    (23, 190, 207): 11,
    (178, 76, 76): 12,
    (247, 182, 210): 13,
    (66, 188, 102): 14,
    (219, 219, 141): 15,
    (140, 57, 197): 16,
    (202, 185, 52): 17,
    (51, 176, 203): 18,
    (200, 54, 131): 19,
    (92, 193, 61): 20,
    (78, 71, 183): 21,
    (172, 114, 82): 22,
    (255, 127, 14): 23,
    (91, 163, 138): 24,
    (153, 98, 156): 25,
    (140, 153, 101): 26,
    (158, 218, 229): 27,
    (100, 125, 154): 28,
    (178, 127, 135): 29,
    (120, 185, 128): 30,
    (146, 111, 194): 31,
    (44, 160, 44): 32,
    (112, 128, 144): 33,
    (96, 207, 209): 34,
    (227, 119, 194): 35,
    (213, 92, 176): 36,
    (94, 106, 211): 37,
    (82, 84, 163): 38,
    (100, 85, 144): 39,
}

color_to_gray = {
    (0, 0, 0): 0,
    (4, 4, 4): 1,
    (3, 3, 3): 2,
    (8, 8, 8): 3,
}

def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

# Function to convert image to grayscale using the defined mapping
def convert_to_grayscale(image):

    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    # Create an output image with the same shape as the input, but single-channel
    grayscale_image = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Iterate over each pixel in the image
    for color, gray_value in color_to_gray.items():
        mask = np.all(image == color, axis=-1)
        grayscale_image[mask] = gray_value
        
    #cv2.imshow("test", grayscale_image)
    #cv2.waitKey(1000)
    # Save the grayscale image
    return grayscale_image
    

class SemanticHypersim(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        
        self.mode = mode
        self.size = size
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        classes_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        #image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
        #cv2.imshow("test.png", image)
        #cv2.waitKey(1000)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        classes = cv2.imread(classes_path)
        classes = cv2.cvtColor(classes, cv2.COLOR_BGR2RGB)
        classes = convert_to_grayscale(classes)
        
        sample = self.transform({'image': image, 'semseg_mask': classes})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['semseg_mask'] = torch.from_numpy(sample['semseg_mask'])
        
        sample['valid_mask'] = (torch.isnan(sample['semseg_mask']) == 0)
        sample['semseg_mask'][sample['valid_mask'] == 0] = 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
