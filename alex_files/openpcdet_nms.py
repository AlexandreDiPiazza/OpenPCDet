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


def perform_nms_on_openpcdet_predictions(
    prediction_files_paths, frame_ids=None, output_path=None
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

    # calib_info = kitti_class.get_calib_info(calib)
    box_preds_lst = []
    cls_score_preds_lst = []
    classes_preds_lst = []
    classes_idx_pred_lst = []
    # min_angles = []
    # max_angles = []
    for prediction_file_path in prediction_files_paths:

        data_numeric = np.genfromtxt(prediction_file_path, usecols=range(9))

        # Boxes are the first 6 columns, reshape to (N, 6)
        bboxes = data_numeric[:, :7]
        # min_angles.append(np.min(data_numeric[:, 6]))
        # max_angles.append(np.max(data_numeric[:, 6]))
        # Scores are the last column, reshape to (N,)
        scores = data_numeric[:, -1]

        classes = np.genfromtxt(prediction_file_path, usecols=7, dtype="str")

        classes_preds_lst.append(classes)
        # classes_idx_pred_lst.append(classes_idx)
        box_preds_lst.append(bboxes)
        cls_score_preds_lst.append(scores)
    # print('Min/Max angles for this file:', np.min(min_angles), np.max(max_angles))
    if len(classes_preds_lst) > 0:
        # At least one prediction
        # TODO We need the tensors on CUDA
        # classes_idx_preds = torch.tensor(np.concatenate(classes_idx_pred_lst)).cuda()
        box_preds = torch.tensor(
            np.concatenate(box_preds_lst), dtype=torch.float32
        ).cuda()
        cls_score_preds = torch.tensor(
            np.concatenate(cls_score_preds_lst), dtype=torch.float32
        ).cuda()
        classes_preds = np.concatenate(classes_preds_lst)

        # print('Box Scores:', cls_score_preds.shape, cls_score_preds)
        # print('Box Preds:', box_preds.shape, box_preds)
        # print("Record Dict Before NMS: ", cls_score_preds, box_preds)
        selected, selected_scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls_score_preds,
            box_preds=box_preds,
            nms_config=nms_config,
            score_thresh=SCORE_THRESH,
        )

        final_scores = selected_scores.cpu().numpy()
        # final_labels = classes_idx_preds[selected]
        final_boxes = box_preds[selected].cpu().numpy()
        final_classes = classes_preds[selected.cpu().numpy()]

        record_dict = {
            "pred_boxes": final_boxes,
            "pred_scores": final_scores,
            "pred_labels": final_classes,
        }
        # print('Final boxes:', final_boxes.shape)
        # print('Final scores:', final_scores.shape)
        # print('Final classes:', final_classes.shape)
        # Save it as a txt file
        output_file_path = output_path / ("%s.txt" % frame_ids[0])
        # print('output file:', output_file_path)
        # Write to the file
        with open(output_file_path, "w") as file:
            for bbox, cls, score in zip(final_boxes, final_classes, final_scores):
                # Format the line to write
                line = (
                    " ".join(map(str, bbox)) + " " + str(cls) + " " + str(score) + "\n"
                )
                file.write(line)
        return record_dict
    else:
        # todo DIP -> this only works with batch_size 1
        # Create an empty text file
        open(output_path / ("%s.txt" % frame_ids[0]), "w").close()

        return None


if __name__ == "__main__":
    # Where to save the new predictions (after ensembling)
    output_path = Path("ensembling_results/aisin")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Path of the folders containing predictions in txt files with KITTI format
    model_predictions_1 = "/home/ubuntu/alex/OpenPCDet/output/cfgs/custom_models/second/default/eval/epoch_30/val/default/final_result/data"
    model_predictions_2 = "/home/ubuntu/alex/OpenPCDet/output/cfgs/custom_models/second/default/eval/epoch_30/val/default/final_result/data"
    model_predictions_3 = "/home/ubuntu/alex/OpenPCDet/output/cfgs/custom_models/second/default/eval/epoch_30/val/default/final_result/data"

    # TODO: most of the code is based on code that can do the calculation by batch -> could be optimized by doing it by batch
    # Iterate through each folder
    for file_name in os.listdir(model_predictions_2):
        print("Doing filename", file_name)
        prediction_files_paths = [
            os.path.join(model_predictions_1, file_name),
            os.path.join(model_predictions_2, file_name),
            os.path.join(model_predictions_3, file_name),
        ]

        after_nms_labels = perform_nms_on_openpcdet_predictions(
            prediction_files_paths,
            frame_ids=[file_name.rstrip(".txt")],
            output_path=output_path,
        )
