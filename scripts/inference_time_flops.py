import time

import cv2
import numpy as np
import torch
from calflops import calculate_flops

import depth_pro

if __name__ == '__main__':

    for i in range(6, 25):
        img_size = i * 64

        start = time.time()
        model, transforms = depth_pro.create_model_and_transforms(img_size=img_size)
        stop = time.time()
        print(f'Model with img_size {img_size} is initialized in {stop - start:.02f} s.')
        model.eval()
        model.cuda()

        pred_times = []

        with torch.no_grad():
            for _ in range(11):
                x, _, f_px = depth_pro.load_rgb('data/rear.jpg')
                x = transforms(x)
                x = x.cuda()
                x = torch.nn.functional.interpolate(
                    x.unsqueeze(0),
                    size=(img_size, img_size),
                    align_corners=False,
                    mode='bilinear',
                )

                start = time.time()
                prediction = model.infer(x, f_px=f_px)
                stop = time.time()
                pred_times.append(stop - start)

            print(f'Prediction {img_size} in {sum(pred_times[1:]) / (len(pred_times) - 1):.02f} s.')

            depth = prediction["depth"]  # Depth in [m].
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.
            print(focallength_px)

            inverse_depth = 1 / depth
            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )

            cv2.imwrite(f'infer_{img_size}.png', (inverse_depth_normalized.cpu().numpy() * 255).astype(np.uint8))

            flops, macs, params = calculate_flops(
                model=model,
                input_shape=(1, 3, img_size, img_size),
                print_results=False,
                print_detailed=False,
            )
            print(f"FLOPs {img_size}: {flops}   MACs:{macs}   Params:{params} \n")
