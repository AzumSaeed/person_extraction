import argparse
import time
from pathlib import Path

import cv2
import os
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from detecTrack.yolo5.utils.datasets import LoadStreams, LoadImages, letterbox
from detecTrack.yolo5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from detecTrack.yolo5.utils.plots import plot_one_box
from detecTrack.yolo5.utils.torch_utils import select_device, load_classifier, time_synchronized


dir_path = os.path.dirname(os.path.realpath(__file__))
device = select_device('cpu')
model = attempt_load(os.path.join(dir_path, "yolov5s.pt"), map_location=device)

def detect(imgframe, confthread=0.25, iouthread=0.35, autonms=True, classes=None,
           imgsz=640, save_img=False):
    # Initialize
    set_logging()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    img = letterbox(imgframe, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, confthread, iouthread, classes=classes, agnostic=autonms)
    t2 = time_synchronized()

    # Apply Classifier
    im0 = imgframe.copy()
    if classify:
        pred = apply_classifier(pred, modelc, img, im0)

    detectlistboxes = []
    detectlistconfs = []
    detectlistcntrs = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if int(cls) != 0:
                    continue
                x, y, w, h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                x, y, w, h = int(x*imgframe.shape[1]), int(y*imgframe.shape[0]), int(w*imgframe.shape[1]), int(h*imgframe.shape[0])
                left = max(0, x - int(w / 2))
                top = max(0, y - int(h / 2))

                conf = float(conf)
                label = '%s' % (names[int(cls)])
                detectlistboxes.append([left, top, w, h])
                detectlistconfs.append(conf)
                detectlistcntrs.append((int(x), int(y)))
                # detectlist.append([label, conf, [left, top, w, h]])

        # Stream results
    indices = cv2.dnn.NMSBoxes(detectlistboxes, detectlistconfs, confthread, iouthread)
    z_box = [detectlistboxes[i[0]] for i in indices]
    picked_score = [detectlistconfs[i[0]] for i in indices]
    z_cntr = [detectlistcntrs[i[0]] for i in indices]
    return z_box, picked_score, z_cntr



if __name__ == '__main__':
    ss = cv2.imread('./images/bus.jpg')
    prddata = detect(ss)
    for nn, ii, pp in prddata:
        x, y, w, h = pp
        ss = cv2.rectangle(ss, (x,y),(x+w,y+h),(0,0,255),3)
        ss = cv2.putText(ss, nn, (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    cv2.imshow("Result", ss)
    cv2.waitKey()
