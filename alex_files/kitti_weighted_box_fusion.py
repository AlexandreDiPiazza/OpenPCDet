import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.append("../")

from pcdet.models.model_utils import model_nms_utils
from kitti_format_to_openpcdet_format import dip_KittiDataset
"""
Peform Weighted Box Fusion between multipled prediction to ensemble the predictions.
We implemented to work in this context - far from optimal implementation

Algorithm paper: https://arxiv.org/pdf/1910.13302
2D implementation: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
Blog: https://learnopencv.com/weighted-boxes-fusion/
"""


def pair_wise_3d_iou_matrix(boxes):
    """
    Computes the pairwise IoU for a set of bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of shape (N, 7) containing N bounding boxes.

    Returns:
        numpy.ndarray: Array of shape (N, N) containing pairwise IoU values.
    """
    N = boxes.shape[0]
    iou_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # TODO: find the IoU function here
            iou = IoU(boxes[i], boxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou  # IoU is symmetric

    return iou_matrix

def perform_weighted_fusion(
    prediction_files_paths, calibration_file_path, frame_ids=None, output_path=None, iou_threshold=0.55
):

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

        # TODO We need the tensors on CUDA
        classes_idx_preds = torch.tensor(np.concatenate(classes_idx_pred_lst)).cuda()
        box_preds = torch.tensor(
            np.concatenate(box_preds_lst), dtype=torch.float32
        ).cuda()
        cls_score_preds = torch.tensor(
            np.concatenate(cls_score_preds_lst), dtype=torch.float32
        ).cuda()

        # TODO: this is highly inneficient, but ok bc actually very few bbox per image (as
        # TODO: ensemble is done post-nms)
        # Compute the NxN IoU matrix (pair-wise IoU)
        iou_matrix = pair_wise_3d_iou_matrix(box_preds)
        N = box_preds.shape[0]
        visited = [False] * N
        clusters = []

        for i in range(N):
            if not visited[i]:
                # Start a new cluster
                cluster = [i]
                visited[i] = True
                # Use a stack to perform depth-first search (DFS)
                stack = [i]

                while stack:
                    current = stack.pop()
                    for j in range(N):
                        if not visited[j] and iou_matrix[current, j] > iou_threshold:
                            visited[j] = True
                            stack.append(j)
                            cluster.append(j)

                clusters.append(cluster)
        # Cluster outputs the indices of the bounding boxes corresponding to the same cluste


        # Pick new bboox of each cluster

        # Pick the new confidence score of each cluster

        # Pick the label of each cluster
        # TODO: Maybe we should compute the cluster per class to make sure we have the same label in a cluster?
        pass
    else:
        # todo DIP -> this only works with batch_size 1
        # Create an empty text file
        open(output_path / ("%s.txt" % frame_ids[0]), "w").close()


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

        after_nms_labels = perform_weighted_fusion(
            prediction_files_paths,
            calibration_file_path,
            frame_ids=[file_name.rstrip(".txt")],
            output_path=output_path,
        )