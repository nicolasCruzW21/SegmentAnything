import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

gray_to_color = {
    0: (128, 128, 128),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (0,255, 255),
    4: (0, 0, 0),  # Default to black if grayscale value not found
}

def convert_to_rgb(grayscale_image):
    if grayscale_image is None:
        raise ValueError("Grayscale image could not be read.")
    
    # Create an output image with the same shape as the input but with three channels
    rgb_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)
    
    # Iterate over gray-to-color mappings and apply them to the output image
    for gray_value, color in gray_to_color.items():
        mask = grayscale_image == gray_value
        rgb_image[mask] = color
    
    return rgb_image
    
def overlay_images(raw_frame, depth, alpha=0.5):
    """
    Overlay the depth prediction on the raw frame.

    Args:
        raw_frame (np.ndarray): The original frame.
        depth (np.ndarray): The depth prediction image.
        alpha (float): The blending factor for the overlay.

    Returns:
        np.ndarray: The combined image with depth overlay.
    """
    
    # Overlay the images
    overlay = cv2.addWeighted(raw_frame, alpha, depth, 1 - alpha, 0)
    return overlay
    
def load_model(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create a new state_dict without the 'module.' prefix if necessary
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # strip the 'module.' prefix
        else:
            new_state_dict[k] = v
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=350)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    # Load the model
    depth_anything = load_model(depth_anything, args.load_from)
    depth_anything = depth_anything.to(DEVICE).eval()
    #if args.pretrained_from:
    #depth_anything.load_state_dict({k: v for k, v in torch.load(args.load_from, map_location='cpu').items()}, strict=False)
    #depth_anything = torch.load(args.load_from)
    
    
    #depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    # Load the checkpoint
    #checkpoint = torch.load(args.load_from, map_location='cpu')

    # Create an empty dictionary to hold the new state dict
    #my_state_dict = {}

    # Adjust the keys in the checkpoint's state dict
    #for key in checkpoint.keys():
    #    new_key = key.replace('module.', '')  # Remove 'module.' prefix
    #    my_state_dict[new_key] = checkpoint[key]

    # Load the adjusted state dict into the model
    #depth_anything.load_state_dict(my_state_dict, strict=False)
    #depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        gray_classes = depth_anything.infer_image(raw_image, args.input_size)
        
        rgb_classes = convert_to_rgb(gray_classes)
        overlay = overlay_images(raw_image, rgb_classes, 0.6)
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_rgb_classes.npy')
            np.save(output_path, rgb_classes)
        
            rgb_classes = (rgb_classes)[:, :, ::-1].astype(np.uint8)
            
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, rgb_classes)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, overlay])
            
            cv2.imwrite(output_path, combined_result)
