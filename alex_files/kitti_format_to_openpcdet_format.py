import sys
import numpy as np

from pathlib import Path

sys.path.append("../")


from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

KITTI_CONVERSION = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
IMAGE_SHAPE = [375, 1212]


class dip_KittiDataset:
    def __init__(self, root_split_path=None):
        self.root_split_path = root_split_path

    def get_calib(self, idx):
        calib_file = self.root_split_path / "calib" / ("%s.txt" % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_calib_from_full_path(self, full_path):
        assert full_path.exists()
        return calibration_kitti.Calibration(full_path)

    def get_calib_info(self, calib):
        P2 = np.concatenate([calib.P2, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        R0_4x4[3, 3] = 1.0
        R0_4x4[:3, :3] = calib.R0
        V2C_4x4 = np.concatenate([calib.V2C, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        calib_info = {"P2": P2, "R0_rect": R0_4x4, "Tr_velo_to_cam": V2C_4x4}
        return calib_info

    def get_label(self, idx):
        label_file = self.root_split_path / "label_2" / ("%s.txt" % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_label_from_full_path(self, full_path):
        assert full_path.exists(), full_path
        return object3d_kitti.get_objects_from_label(full_path)

    def convert_kitti_format_to_openPCDet_format(
        self, calib, obj_list, score_predicted=False
    ):
        """
        Transform from the KITTI format to the format excpected by OpenPCDet.
        Most of the code is taken from: pcdet/datasets/kitti/kitti_dataset.py in the get_infos function.
        (https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/kitti/kitti_dataset.py)

        KITTI FORMAT -> PCDET FORMAT

        KITTI FORMAT:
        15 values: #truncated#occluded#alpha#bbox#dimensions#location#rotation_y#score
        - **type**:  1 value ; Describes the type of object: eg  'Car', 'Van’
        - **truncated:** 1 value; Float from 0 (non-truncated) to 1 (truncated), where
        truncated refers to the object leaving image boundaries. ( used with occlusion to set object as easy/moderate/hard)
        - **occluded**: 1 value; Integer (0,1,2,3) indicating occlusion state:
        0 = fully visible, 1 = partly occluded
        2 = largely occluded, 3 = unknown. - ( used with truncated to define object as easy/medium/hard)
        - **alpha:** 1 value;  Observation angle of object, ranging [-pi..pi].  It is the angle between the object's heading direction and the positive x-axis of the camera
        - **bbox:** 4 values **;** 2D bounding box of object in the image (0-based index):
        contains left, top, right, bottom pixel coordinates
        - **dimensions**: 3 values ; 3D object dimensions: height, width, length (in meters)
        - **location:** 3 values**;** 3D object location x,y,z in camera coordinates (in meters)
        - **rotation_y:** 1 value  Rotation ry around Y-axis in camera coordinates [-pi..pi].  The rotation of the object around the y-axis in camera coordinates, in radians

        PCDET FORMAT:
        # format: [x y z dx dy dz heading_angle category_name score[Optional]]

        args:
            calib: calibration_kitti.Calibration
            obj_list: list[object3d_kitti.Object3D]

        Returns:
            gt_boxes_lidar: np.ndarray: Shape N,7 or N,8 (if we have score) - N is the nbr of object predicted in the file
        """

        # KITTI Format
        annotations = {}
        annotations["name"] = np.array([obj.cls_type for obj in obj_list])
        annotations["truncated"] = np.array([obj.truncation for obj in obj_list])
        annotations["occluded"] = np.array([obj.occlusion for obj in obj_list])
        annotations["alpha"] = np.array([obj.alpha for obj in obj_list])
        annotations["bbox"] = np.concatenate(
            [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0
        )
        annotations["dimensions"] = np.array(
            [[obj.l, obj.h, obj.w] for obj in obj_list]
        )  # lhw(camera) format
        annotations["location"] = np.concatenate(
            [obj.loc.reshape(1, 3) for obj in obj_list], axis=0
        )
        annotations["rotation_y"] = np.array([obj.ry for obj in obj_list])
        annotations["score"] = np.array([obj.score for obj in obj_list])
        annotations["difficulty"] = np.array([obj.level for obj in obj_list], np.int32)

        # print("Annotations: ", annotations)

        num_objects = len(
            [obj.cls_type for obj in obj_list if obj.cls_type != "DontCare"]
        )
        num_gt = len(annotations["name"])
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations["index"] = np.array(index, dtype=np.int32)

        # KITTI Format -> OpenPCDet Format
        loc = annotations["location"][:num_objects]
        dims = annotations["dimensions"][:num_objects]
        rots = annotations["rotation_y"][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate(
            [loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1
        )

        annotations["gt_boxes_lidar"] = gt_boxes_lidar
        gt_indexes = [KITTI_CONVERSION[cat] for cat in annotations["name"]]

        return annotations["name"], gt_indexes, gt_boxes_lidar, annotations["score"]

    @staticmethod
    def convert_OpenPCDet_to_kitti_format(
        pred_dicts, calib, frame_ids, output_path, class_names=None, image_shape=None
    ):

        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "truncated": np.zeros(num_samples),
                "occluded": np.zeros(num_samples),
                "alpha": np.zeros(num_samples),
                "bbox": np.zeros([num_samples, 4]),
                "dimensions": np.zeros([num_samples, 3]),
                "location": np.zeros([num_samples, 3]),
                "rotation_y": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros([num_samples, 7]),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict, calib, class_names, image_shape):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            # print('Pred boxes', pred_boxes)
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, calib
            )
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            # print('Class names', class_names)
            # print('Pred labels', pred_labels)
            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["alpha"] = (
                -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0])
                + pred_boxes_camera[:, 6]
            )
            pred_dict["bbox"] = pred_boxes_img
            pred_dict["dimensions"] = pred_boxes_camera[:, 3:6]
            pred_dict["location"] = pred_boxes_camera[:, 0:3]
            pred_dict["rotation_y"] = pred_boxes_camera[:, 6]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = frame_ids[index]

            # single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict = generate_single_sample_dict(
                box_dict,
                calib,
                class_names=list(KITTI_CONVERSION.keys()),
                image_shape=IMAGE_SHAPE,
            )
            single_pred_dict["frame_id"] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ("%s.txt" % frame_id)
                with open(cur_det_file, "w") as f:
                    bbox = single_pred_dict["bbox"]
                    loc = single_pred_dict["location"]
                    dims = single_pred_dict["dimensions"]  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            "%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
                            % (
                                single_pred_dict["name"][idx],
                                single_pred_dict["alpha"][idx],
                                bbox[idx][0],
                                bbox[idx][1],
                                bbox[idx][2],
                                bbox[idx][3],
                                dims[idx][1],
                                dims[idx][2],
                                dims[idx][0],
                                loc[idx][0],
                                loc[idx][1],
                                loc[idx][2],
                                single_pred_dict["rotation_y"][idx],
                                single_pred_dict["score"][idx],
                            ),
                            file=f,
                        )

        return annos


if __name__ == "__main__":
    root_split_path = Path("../data/kitti/training/")

    kitti_class = dip_KittiDataset(root_split_path)
    calib_path = Path(
        "/home/ubuntu/alex/OpenPCDet/data/kitti/training/calib/000000.txt"
    )
    label_path = Path(
        "/home/ubuntu/alex/OpenPCDet/output/cfgs/kitti_models/pv_rcnn/default/eval/epoch_80/val/default/final_result/data/000000.txt"
    )
    calib = kitti_class.get_calib_from_full_path(calib_path)
    kitti_obj_list = kitti_class.get_label_from_full_path(label_path)

    classes, class_idx, bboxes, scores = (
        kitti_class.convert_kitti_format_to_openPCDet_format(
            calib, kitti_obj_list, score_predicted=True
        )
    )
    print("Classes", classes)
    print("Class idx", class_idx)
    print("Bbox ", bboxes)
    print("Scores", scores)
