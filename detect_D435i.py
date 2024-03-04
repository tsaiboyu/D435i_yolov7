import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
#%%
pipeline = rs.pipeline()  # 定義流程pipeline
config = rs.config()  # 定義配置config
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程開始
align_to = rs.stream.color  
align = rs.align(align_to)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待獲取影像幀
    aligned_frames = align.process(frames)  # 取得對齊幀
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 取得對齊幀中的depth幀
    color_frame = aligned_frames.get_color_frame()  # 取得對齊幀中的color幀

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 取得相機內參
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 取得深度參數（像素座標系轉相機座標系會用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度圖（默認16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度圖（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度圖
    color_image = np.asanyarray(color_frame.get_data())  # RGB圖

    # 返回相機内参、深度參數、彩色圖、深度圖、齊幀中的depth幀
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame



#%%
def detect(save_img=False):
    #檢測設定
    trace = True
    device = ''
    classes = None
    agnostic_nms = False
    augment = False
    weights = 'yolov7.pt'
    imgsz = 640
    conf_thres = 0.5
    iou_thres = 0.5
    half = True # 用GPU = True  
    #
    
    device = select_device(device)
    print(f"device : Using {device}")
    
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    
    
    
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    while True:
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        if not depth_image.any() or not color_image.any():
            continue
        im0 = color_image.copy()
        # Padded resize
        img = letterbox(color_image, imgsz, stride=stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        #
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()
	# Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                camera_xyz_list=[]
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}'
                    ux = int((int(xyxy[0]) + int(xyxy[2]))/2)
                    uy = int((int(xyxy[1]) + int(xyxy[3]))/2)
                    dis = aligned_depth_frame.get_distance(ux, uy)
                    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                    camera_xyz = camera_xyz.tolist()
                    cv2.circle(im0, (ux,uy), 4, (255, 255, 255), 5)#标出中心点
                    cv2.putText(im0, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标
                    #camera_xyz_list.append(camera_xyz)
                    
                    # Add bbox to image
                    #label = f'{names[int(cls)]} {conf:.2f}'                    
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    #c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    #print(f"左上角座標 : {c1}\n 右下角座標 : {c2}")
        cv2.imshow('result', im0)
        keycode = cv2.waitKeyEx(1)
        if keycode == 27: #按下Esc退出
            print('Break.')
            break
            # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    print('detect start')
    with torch.no_grad():
        detect()


