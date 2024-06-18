import os
import sys
import numpy as np
import torch
from pathlib import Path
from easydict import EasyDict

sys.path.append("../")

from pcdet.models.model_utils import model_nms_utils
from kitti_format_to_openpcdet_format import dip_KittiDataset


"""
Perform NMS between multiple prediction folders. Each folder is the predictions of a model, and contains the predictions in txt files in the KITTI format
"""

# TODO: Right now, we need the tensors to be on cuda


def perform_nms_on_kitti_predictions(
    prediction_files_paths, calibration_file_path, frame_ids=None, output_path=None
):
    # TODO: can be done in a batch instead of image by image
    nms_config = EasyDict(
        {
            "NMS_TYPE": "nms_gpu",
            "MULTI_CLASSES_NMS": False,
            "NMS_PRE_MAXSIZE": 1024,
            "NMS_POST_MAXSIZE": 100,
            "NMS_THRESH": 0.7,
        }
    )
    SCORE_THRESH = 0.1

    kitti_class = dip_KittiDataset()
    # Calibration file
    calib = kitti_class.get_calib_from_full_path(Path(calibration_file_path))
    # calib_info = kitti_class.get_calib_info(calib)
    box_preds_lst = []
    cls_score_preds_lst = []
    classes_preds_lst = []
    classes_idx_pred_lst = []
    for prediction_file_path in prediction_files_paths:
        predictions_kitti = kitti_class.get_label_from_full_path(
            Path(prediction_file_path)
        )
        if len(predictions_kitti) == 0:
            break
        classes, classes_idx, bboxes, scores = (
            kitti_class.convert_kitti_format_to_openPCDet_format(
                calib, predictions_kitti, score_predicted=True
            )
        )
        classes_preds_lst.append(classes)
        classes_idx_pred_lst.append(classes_idx)
        box_preds_lst.append(bboxes)
        cls_score_preds_lst.append(scores)

    if len(classes_idx_pred_lst) > 0:
        # At least one prediction
        # TODO We need the tensors on CUDA
        classes_idx_preds = torch.tensor(np.concatenate(classes_idx_pred_lst)).cuda()
        box_preds = torch.tensor(
            np.concatenate(box_preds_lst), dtype=torch.float32
        ).cuda()
        cls_score_preds = torch.tensor(
            np.concatenate(cls_score_preds_lst), dtype=torch.float32
        ).cuda()

        # print("Record Dict Before NMS: ", cls_score_preds, box_preds)
        selected, selected_scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls_score_preds,
            box_preds=box_preds,
            nms_config=nms_config,
            score_thresh=SCORE_THRESH,
        )

        final_scores = selected_scores
        final_labels = classes_idx_preds[selected]
        final_boxes = box_preds[selected]
        record_dict = {
            "pred_boxes": final_boxes,
            "pred_scores": final_scores,
            "pred_labels": final_labels,
        }
        # Convert back to KITTI format
        kitti_class = kitti_class.convert_OpenPCDet_to_kitti_format(
            pred_dicts=[record_dict],
            calib=calib,
            frame_ids=frame_ids,
            output_path=output_path,
        )
        return record_dict
    else:
        # todo DIP -> this only works with batch_size 1
        # Create an empty text file
        open(output_path / ("%s.txt" % frame_ids[0]), "w").close()

        return None


if __name__ == "__main__":
    # Path of the calibration files
    kitti_calib_path = "../data/kitti/training/calib"

    # Where to save the new predictions (after ensembling)
    output_path = Path("ensembling_results/label_2")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Path of the folders containing predictions in txt files with KITTI format
    model_predictions_1 = "/home/ubuntu/alex/OpenPCDet/output/cfgs/kitti_models/pv_rcnn/default/eval/epoch_80/val/default/final_result/data/"
    model_predictions_2 = "../output/cfgs/kitti_models/second/default/eval/epoch_80/val/default/final_result/data"
    model_predictions_3 = "../output/cfgs/kitti_models/pointrcnn_iou/default/eval/epoch_80/val/default/final_result/data"

    # TODO: most of the code is based on code that can do the calculation by batch -> could be optimized by doing it by batch
    # Iterate through each folder
    for file_name in os.listdir(model_predictions_2):
        # if os.path.isfile(os.path.join(model_predictions_1, file_name)):
        #     print(file_name)
        #     exit()
        print("Doing filename", file_name)
        prediction_files_paths = [
            os.path.join(model_predictions_1, file_name),
            os.path.join(model_predictions_2, file_name),
            os.path.join(model_predictions_3, file_name),
        ]
        calibration_file_path = os.path.join(kitti_calib_path, file_name)

        after_nms_labels = perform_nms_on_kitti_predictions(
            prediction_files_paths,
            calibration_file_path,
            frame_ids=[file_name.rstrip(".txt")],
            output_path=output_path,
        )
