import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

gray_to_color = {
    0: (255, 0, 0),
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {k[7:]: v for k, v in checkpoint.items() if k.startswith('module.')}
    new_state_dict.update({k: v for k, v in checkpoint.items() if not k.startswith('module.')})
    model.load_state_dict(new_state_dict, strict=False)
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video or directory of videos')
    parser.add_argument('--input-size', type=int, default=350, help='Input size for depth estimation')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory for processed videos')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Model encoder type')
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', help='Path to the model checkpoint')
    parser.add_argument('--max-depth', type=float, default=20, help='Maximum depth for normalization')
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='Save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='Do not apply a colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything = load_model(depth_anything, args.load_from)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('.txt'):
            with open(args.video_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        if not os.path.isdir(args.video_path):
            raise FileNotFoundError(f"Directory {args.video_path} does not exist.")
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    
    for k, filename in enumerate(filenames):
        print(f'Processing file {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        if not raw_video.isOpened():
            print(f"Error: Could not open video file {filename}.")
            continue
        
        frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            gray_classes = depth_anything.infer_image(raw_frame, args.input_size)
            #unique_depth_values = np.unique(gray_classes)
            #print("Unique depth values:", unique_depth_values)
            rgb_classes = convert_to_rgb(gray_classes)
            overlay = overlay_images(raw_frame, rgb_classes, 0.6)
            cv2.imshow("test", overlay)
            cv2.waitKey(10)
            
            if args.grayscale:
                depth = np.repeat(rgb_classes[..., np.newaxis], 3, axis=-1)
            else:
                depth = rgb_classes
            
            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, overlay])
                out.write(combined_frame)

        raw_video.release()
        out.release()

