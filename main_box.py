
import os
import cv2
import json
import numpy as np
import pandas as pd

import torch

from ultralytics import YOLO
from ultralytics.utils import set_logging
from ultralytics.utils.ops import scale_masks, scale_boxes

from imgviz.io import label_colormap
from PIL import Image
from pathlib import Path
from skimage import morphology
from scipy.ndimage import distance_transform_edt


##########################################################################
# 利用YOLO26进行病害实例分割，基于像素与真实信息转换关系，得到量化结果并保存为csv

def analyze_crack_mask(binary_mask):
    # 确保掩模是严格的布尔值或 0/1
    mask = (binary_mask > 0).astype(np.uint8)
    # 1. 计算总面积 (像素总数)
    area = np.sum(mask)
    if area == 0:
        return 0, 0, 0

    # 2. 提取骨架
    # skimage 的 skeletonize 接受布尔数组，返回布尔数组
    skeleton = morphology.skeletonize(mask)
    # 近似计算长度 (骨架像素点总数)
    # 若需极高精度，这里需写额外的寻路逻辑区分直线相连和对角相连
    length = np.sum(skeleton)

    # 3. 方法A距离变换求宽度
    # EDT 计算前景中每个点到背景的最短距离
    dist_transform = distance_transform_edt(mask)
    # 提取骨架所在位置的距离值
    skeleton_distances = dist_transform[skeleton]
    # 计算宽度 (距离等于半宽，所以要乘 2)
    max_width = np.max(skeleton_distances)
    avg_width_edt = np.mean(skeleton_distances)

    # 方法B计算的平均宽度
    avg_width_area = area / length if length > 0 else 0

    return {
        "length_pixels": length,
        "max_width_pixels": max_width,
        "avg_width_edt_pixels": avg_width_edt,
        "avg_width_area_pixels": avg_width_area
    }


def predict_ins_csv(checkpoint, images_path, save_fig=False):
    yolo_model = YOLO(model=checkpoint, verbose=True)
    pred_result = yolo_model.predict(source=images_path, save=save_fig, stream=False)
    result_csv = Path(images_path).stem + ".csv"
    all_data = []
    headers = ["image", "type", "location", "information", "interval", "line", "defectID"]

    n = 0
    for i, r in enumerate(pred_result):
        if not r:
            print(f"{Path(r.path).stem} no detected class")
            vals = [Path(r.path).name, None, None, None, None, None, None]
            all_data.append(vals)
        else:
            cls_index = r.boxes.cls.cpu().numpy().astype(np.uint8).tolist()
            cls_names = r.names
            ori_boxes = scale_boxes(img1_shape=r.masks.shape[-2:], boxes=r.boxes.xyxy.cpu().numpy(), img0_shape=r.orig_shape)
            ori_boxes = np.around(ori_boxes, 3)
            ori_masks = scale_masks(masks=r.masks.data[None], shape=r.orig_shape)
            ori_masks = ori_masks.squeeze(0).cpu().numpy().astype(np.uint8)
            for j, per_mask in enumerate(ori_masks):
                per_defect = cls_names[cls_index[j]]
                per_location = ori_boxes[j].tolist()

                if per_defect in {"leakage", "spall"}:
                    pixel_area = np.count_nonzero(per_mask)
                    per_info = round(pixel_area * 0.0446, 3)
                else:
                    per_info = analyze_crack_mask(per_mask)
                    per_info = round(per_info["avg_width_edt_pixels"] * 0.2112, 3)

                vals = [Path(r.path).name, per_defect, per_location, per_info, Path(r.path).name.split("_")[1],
                        Path(r.path).name.split("_")[0], n+1]
                all_data.append(vals)
                n += 1

    predict_df = pd.DataFrame(all_data)
    # result_path = Path(images_path).parent / result_csv
    predict_df.to_csv(result_csv, index=False, header=headers)
##########################################################################


if __name__ == '__main__':
    print(torch.cuda.is_available())
    # 模型训练
    # model = YOLO(model="yolo26n-seg.pt", verbose=True)
    # train_result = model.train(data="original_data.yaml", epochs=100, imgsz=640, batch=8,
    #                            project="segment", name="yolo26n_train")

    # 模型预测/评估
    # model = YOLO(model="best.pt", verbose=True)
    # pred_results = model.predict(source="test", save=True, project="YOLO_predict", name="test")
    # val_result = model.val(data="original_data.yaml")

    # 模型预测结果存储
    predict_ins_csv(checkpoint="best.pt",
                    images_path="image", save_fig=False)


