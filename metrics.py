import cv2
import os
import numpy as np
from matplotlib import pyplot as plt




def get_iou(root):
    class_wise_dict = {}
    for ech in os.listdir(root):
        if ech.startswith('.'):
            continue
        gt_cls = ech.split('_')[0]

        cls_lst = class_wise_dict.get(gt_cls,list())
        cls_lst.append(os.path.join(root,ech))
        class_wise_dict[gt_cls] = cls_lst

    class_wise_dict = {}
    for ech in os.listdir(root):
        if ech.startswith('.'):
            continue
        gt_cls = ech.split('_')[0]

        cls_lst = class_wise_dict.get(gt_cls,list())
        cls_lst.append(os.path.join(root,ech))
        class_wise_dict[gt_cls] = cls_lst


    iou_dict = {}
    for ech in class_wise_dict.keys():
        for ech_img in class_wise_dict[ech]:
            if "_gt_" in ech_img:
                continue
            pd = cv2.imread(ech_img)
            gt = cv2.imread(ech_img.replace('_pd_','_gt_'))

            intersection = np.logical_and(pd, gt)
            union = np.logical_or(pd, gt)
            iou_score = np.sum(intersection) / np.sum(union)

            iou_lists = iou_dict.get(ech,[])
            iou_lists.append(iou_score)
            iou_dict[ech] = iou_lists

    total_avg = []
    for ech in iou_dict:
        cur_avg = np.average(iou_dict[ech])
        print(ech, cur_avg)
        total_avg.append(cur_avg)
    print('iou : ', np.average(cur_avg))

if __name__ == "__main__":
    root = "./visualize_iou_inference/"
    get_iou(root)