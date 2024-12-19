import time

import cv2
import numpy as np
import torch
from calflops import calculate_flops

import depth_pro

if __name__ == '__main__':
    start = time.time()
    model, transforms = depth_pro.create_model_and_transforms(img_size=384)
    stop = time.time()
    print(f'Model is initialized in {stop - start:.02f} s.')

    with torch.no_grad():
        model.eval()
        model.cuda()

        x, _, f_px = depth_pro.load_rgb('data/rear.jpg')
        x = transforms(x)
        x = x.cuda()

        start = time.time()
        prediction = model.infer(x, f_px=f_px)
        stop = time.time()
        print(f'Prediction in {stop - start:.02f} s.')

        depth = prediction["depth"]  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        cv2.imwrite('mod_384.png', (inverse_depth_normalized.cpu().numpy() * 255).astype(np.uint8))

        flops, macs, params = calculate_flops(
            model=model,
            input_shape=(1, 3, 384, 384),
        )
        print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))