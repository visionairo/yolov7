# By: Jeroen Hoogers. Adaptation of the detect.py script

import json
#from PIL import Image
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import os
import logging
import argparse
from dataclasses import dataclass
import numpy as np
import math
import glob
import torch
import time

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Load a pretrained YOLO11n model
# model = YOLO("YOLOv7-X.pt", task="detect", verbose=True)
# model.info()

IMG_WIDTH = 960
IMG_HEIGHT = 720
YOLO_PREFIX = "yolorawoutput_"
JPG_PATTERN = "rgb_%08d.jpg"
logging.basicConfig(level=logging.WARNING)

MODEL = "models/mesquite-yolov7.pt"

set_logging()
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
imgsz = 640
model = attempt_load(MODEL, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

# if trace:
#     model = TracedModel(model, device, opt.img_size)

if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

def detect(path):
    results = {}

    dataset = LoadImages(path, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            detections = []

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    bbox = {"cx": xywh[0], "cy": xywh[1] , "w": xywh[2], "h": xywh[3]}
                    detection = {"bbox": bbox, "confidence": float(conf), "classid": int(cls)}
                    detections.append(detection)

                # # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if save_txt:  # Write to file
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{p}: {s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # print(detections)
            results[p] = detections

#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         #print(f"Results saved to {save_dir}{s}")

    return results

def run(dir):
    cameras = glob.glob(f'{dir}/*/*')
    print(cameras)

    for camera_dir in cameras:
        results = []

        yolo_output = detect(camera_dir)
        frames = glob.glob(f'{camera_dir}/rgb_*.jpg')
        frames.sort()

        lastframe = 0
        for frame in frames:
            frame_idx = int(os.path.splitext(frame)[0][-8:])

            # TODO: run inference
            # print(f"frame: {frame} id: {frame_idx}")

            detections = yolo_output[frame]
            # print(detections)
        
            lastframe = frame_idx
            results.append({"frame" : frame_idx, "detections": detections})
        
        json_yolo = json.dumps({"results" : results})
        json_path = os.path.join(camera_dir, f"{YOLO_PREFIX}00000000.json")
        with open(json_path, "w") as outfile:
            json.dump({"results" : results}, outfile, indent=2)

# def plot_camera(dir):
#     file = os.path.filename(dir).splitext()[0]

#     # create OpenCV video writer
#     video = cv2.VideoWriter(f'{file}.mp4', cv2.VideoWriter_fourcc('A','V','C','1'), 1, (mat.shape[0],mat.shape[1]))


#     # loop over your images
#     for i in xrange(len(img)):

        
#         fig = plt.figure()
#         plt.imshow(img[i], cmap=cm.Greys_r)

#         # put pixel buffer in numpy array
#         canvas = FigureCanvas(fig)
#         canvas.draw()
#         mat = np.array(canvas.renderer._renderer)
#         mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

#         # write frame to video
#         video.write(mat)

#     # close video writer
#     cv2.destroyAllWindows()
#     video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source') 
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    run(opt.source)