import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.append("../")

from pcdet.models.model_utils import model_nms_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import paired_boxes_iou3d_gpu
from kitti_format_to_openpcdet_format import dip_KittiDataset

"""
Peform Weighted Box Fusion between multipled prediction to ensemble the predictions.
We implemented to work in this context - far from optimal implementation

Algorithm paper: https://arxiv.org/pdf/1910.13302
2D implementation: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
Blog: https://learnopencv.com/weighted-boxes-fusion/
"""


CLASS_TO_STRING = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
STRING_TO_CLASS = {v: k for k, v in CLASS_TO_STRING.items()}


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
            iou = paired_boxes_iou3d_gpu(
                torch.unsqueeze(boxes[i], 0), torch.unsqueeze(boxes[j], 0)
            )  # Unsqueeze to have tensors of shape 1,7 as input to the function
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou  # IoU is symmetric

    return iou_matrix


def perform_weighted_fusion(
    prediction_files_paths: list,
    frame_ids: list = None,
    output_path: Path = None,
    iou_threshold: float = 0.5,
    selection_strategy: str = "affirmative",
):
    """
    Pefroms the weighted fusion algorithm:
        - Regroup the bounding boxes that have a high IoU
        - Compute the weighted average of the bounding boxes based on the scores


    For the Selection strategy, based on https://github.com/ancasag/ensembleObjectDetection , we allow 3 different startegies:
        - affirmative: the box is kept if it is present in at least one of the models
        - consensus: the box is kept if it is present in the majority of the models
        - unanimous: the box is kept if it is present in all the models

    Args:
        - selection_strategy: float, the threshold to consider two boxes as the same
        - selectrion_stragegy: str, the strategy to select the boxes. Can be 'affirmative', 'consensus' or 'unanimous'
    Returns:
    """
    assert selection_strategy in [
        "affirmative",
        "consensus",
        "unanimous",
    ], "The selection strategy must be one of 'affirmative', 'consensus' or 'unanimous'"

    ensemble_n = len(prediction_files_paths)  # nbr of models taken from the ensemble

    box_preds_lst = []
    cls_score_preds_lst = []
    classes_preds_lst = []
    classes_idx_pred_lst = []
    for prediction_file_path in prediction_files_paths:

        data_numeric = np.genfromtxt(prediction_file_path, usecols=range(9))

        # Boxes are the first 6 columns, reshape to (N, 6)
        bboxes = data_numeric[:, :7]
        # min_angles.append(np.min(data_numeric[:, 6]))
        # max_angles.append(np.max(data_numeric[:, 6]))
        # Scores are the last column, reshape to (N,)
        scores = data_numeric[:, -1]

        classes_str = np.genfromtxt(prediction_file_path, usecols=7, dtype="str")
        classes_idx = np.vectorize(CLASS_TO_STRING.get)(classes_str)

        classes_preds_lst.append(classes_str)
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
        # Shape N,N, where diagonal are ones.
        iou_matrix: np.ndarray = pair_wise_3d_iou_matrix(box_preds)

        nbr_bbox_predicted = box_preds.shape[0]
        visited = [False] * nbr_bbox_predicted
        clusters = []
        # Cluster outputs the indices of the bounding boxes corresponding to the same clusters
        # Clusters is a list of len K, where K is the nbr of clusters.
        # Each element of the list is another list, that contains the indices corresponding to this cluster
        for i in range(nbr_bbox_predicted):
            if not visited[i]:
                # Start a new cluster
                cluster = [i]
                visited[i] = True
                # Use a stack to perform depth-first search (DFS)
                stack = [i]

                while stack:
                    current = stack.pop()
                    for j in range(nbr_bbox_predicted):
                        if not visited[j] and iou_matrix[current, j] > iou_threshold:
                            visited[j] = True
                            stack.append(j)
                            cluster.append(j)

                clusters.append(cluster)

        final_boxes = []
        final_scores = []
        final_labels = []
        final_labels_str = []
        for cluster in clusters:
            # TODO: Based on the IoU threshold, it might be possible that a cluster has multiple bb of the same model
            # TODO: , in that case, it doenst  make sense to compute the nbr of models as the length of the cluster
            nbr_models_predictions = len(cluster)
            if selection_strategy == "consensus":
                if nbr_models_predictions < ensemble_n / 2:
                    # We have less than half of the models predicting this bbox, we discard it
                    continue
            elif selection_strategy == "unanimous":
                if nbr_models_predictions < ensemble_n:
                    # Not all models predicted this bbox, we discard it
                    continue
            # Use the index_select function to extract rows based on the indices
            cluster_tensor = torch.tensor(cluster).cuda()
            selected_scores = torch.index_select(
                cls_score_preds, 0, cluster_tensor
            )  # Selected Scores
            selected_labels = torch.index_select(
                classes_idx_preds, 0, cluster_tensor
            )  # Selected Labels
            selected_bboxs = torch.index_select(
                box_preds, 0, cluster_tensor
            )  # Selected Labels
            ### Average the bbox
            # Compute the weighted average of bounding boxes based on scores
            # DIP -  I got inspired from https://www.kaggle.com/competitions/3d-object-detection-for-autonomous-vehicles/discussion/122820
            #! #! We cannot directly do the avg of the angle because the angle is modulo 2 pis!!
            sin_values = torch.sin(2 * selected_bboxs[:, 6])
            cos_values = torch.cos(2 * selected_bboxs[:, 6])

            s = torch.sum(selected_scores * sin_values) / torch.sum(selected_scores)
            c = torch.sum(selected_scores * cos_values) / torch.sum(selected_scores)
            mean_angle = torch.atan2(s, c).unsqueeze(0) / 2

            weighted_avg_box_coords = torch.sum(
                selected_bboxs[:, :6] * selected_scores.view(-1, 1), dim=0
            ) / torch.sum(selected_scores)

            # print('mean angle shape', mean_angle)
            # print('mean weighted avg box', weighted_avg_box_coords.shape)

            weighted_avg_box = torch.cat((weighted_avg_box_coords, mean_angle))

            # Reshape the weighted average box to have shape (1, 7)
            weighted_avg_box = weighted_avg_box.unsqueeze(0)

            ### Take most present labels
            # todo - big assumption for now, we assume all the boxes in a cluster have the same labels
            box_label = selected_labels[0]

            ### Compute the new conf score based on the formula of the paper
            box_score = torch.mean(selected_scores)
            T = len(cluster)  # nbr of bbox predicted in this cluser

            box_score = box_score * min(T, ensemble_n) / ensemble_n

            # Append
            final_boxes.append(weighted_avg_box)
            final_scores.append(box_score)
            final_labels.append(box_label)

            # print('Final Values for this cluster: ')
            # print('bbox: ', weighted_avg_box)
            # print('Score', box_score)
            # print('Label', box_label)

        if len(final_boxes) == 0:
            # After the selection strategy, we might have no boxes left
            # Create an empty text file
            open(output_path / ("%s.txt" % frame_ids[0]), "w").close()
        else:
            final_boxes = torch.cat(final_boxes, dim=0).cpu().numpy()
            final_scores = torch.stack(final_scores).cpu().numpy()
            final_labels = torch.stack(final_labels).cpu().numpy()
            # Convert the labels in to strings
            print(final_labels)
            print(final_labels.shape)
            final_labels_str = np.vectorize(STRING_TO_CLASS.get)(final_labels)

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
                "pred_labels_str": final_labels_str,
            }
            output_file_path = output_path / ("%s.txt" % frame_ids[0])
            # print('output file:', output_file_path)
            # Write to the file
            with open(output_file_path, "w") as file:
                for bbox, cls, score in zip(
                    final_boxes, final_labels_str, final_scores
                ):
                    # Format the line to write
                    line = (
                        " ".join(map(str, bbox))
                        + " "
                        + str(cls)
                        + " "
                        + str(score)
                        + "\n"
                    )
                    file.write(line)

    else:
        # todo DIP -> this only works with batch_size 1
        # Create an empty text file
        open(output_path / ("%s.txt" % frame_ids[0]), "w").close()


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
    for file_name in os.listdir(model_predictions_1):
        # if os.path.isfile(os.path.join(model_predictions_1, file_name)):
        #     print(file_name)
        #     exit()
        print("Doing filename", file_name)

        prediction_files_paths = [
            os.path.join(model_predictions_1, file_name),
            os.path.join(model_predictions_2, file_name),
            os.path.join(model_predictions_3, file_name),
        ]

        perform_weighted_fusion(
            prediction_files_paths,
            frame_ids=[file_name.rstrip(".txt")],
            output_path=output_path,
            selection_strategy="unanimous",
        )
