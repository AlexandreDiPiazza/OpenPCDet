
##  1 - Setup the conda venv

My way to setup the venv:


```
conda create -n openpcdet python=3.8
conda activate openpcdet
```

Install pytorch first:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
Install spconv based on CUDA version (usind CUDA > 12.0 is fine, just pick install spconv for 12.0):
```
https://github.com/traveller59/spconv
```

Pip install the requirements:
```
pip install -r requirements.txt
```

Run the setup:
```
python setup.py develop
```

The above command is crucial, but at first probably runs into errors, that can be solved by pip insalling libraries and running the command again, eg:
- Error: "Best match: scikit-image 0.23.2
Processing scikit_image-0.23.2.tar.gz
error: Couldn't find a setup script in /tmp/easy_install-uwt7cz7q/scikit_image-0.23.2.tar.gz"  -> ```pip install scikit-image==0.18.1```
-  Error: "File "/tmp/easy_install-opo696u7/numba-0.60.0rc1/setup.py", line 48, in _guard_py_ver
    # 'spconv',  # spconv has different names depending on the cuda version
RuntimeError: Cannot install on Python version 3.8.19; only versions >=3.9,<3.13 are supported." -> ```pip install numba==0.48```


### Testing models on kitti (validation set):
Note: -> it looks like the test is done on the train + validation set?


Test PointPillar
````
python test.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 6 --ckpt ../checkpoint/kitti/pointpillar_7728.pth --save_to_file --infer_time
````
-> Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.6608, 92.2405, 91.3173
bev  AP:92.0419, 88.0540, 86.6634
3d   AP:87.7520, 78.4039, 75.1921

````
python test.py --cfg_file ./cfgs/kitti_models/pointpillar_pyramid_aug.yaml --batch_size 6 --ckpt ../checkpoint/kitti/pointpillar_7728.pth --save_to_file --infer_time
````
->  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7786, 89.8072, 88.7942
bev  AP:89.6621, 87.1725, 84.3776
3d   AP:86.4617, 77.2840, 74.6621

````
python test.py --cfg_file ./cfgs/kitti_models/pointpillar_newaugs.yaml --batch_size 6 --ckpt ../checkpoint/kitti/pointpillar_7728.pth --save_to_file --infer_time
````
-> Car AP@0.70, 0.70, 0.70:
bbox AP:90.7786, 89.8072, 88.7942
bev  AP:89.6621, 87.1725, 84.3776
3d   AP:86.4617, 77.2840, 74.6621

Test Second

````
python test.py --cfg_file ./cfgs/kitti_models/second.yaml --batch_size 6 --ckpt ../checkpoint/kitti/second_7862.pth --save_to_file --infer_time
````
-> Car AP@0.70, 0.70, 0.70:
bbox AP:90.7803, 89.8999, 89.0440
bev  AP:90.0097, 87.9244, 86.4554
3d   AP:88.5922, 78.6029, 77.1686


Test pv_rcnn

````
python test.py --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml --batch_size 6 --ckpt ../checkpoint/kitti/pv_rcnn_8369.pth --save_to_file --infer_time
````

-> Car AP@0.70, 0.70, 0.70:
bbox AP:96.2423, 89.4946, 89.2401
bev  AP:90.0946, 87.8943, 87.4082
3d   AP:89.3325, 83.6905, 78.7183



### Training models on kitti:
Train PointPillar
```
python train.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --batch_size 6 --save_to_file
```

Train pv_rcnn
```
python train.py --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml --batch_size 6 --save_to_file
```

## Training on custom dataset:

```
python train.py --cfg_file ./cfgs/custom_models/pv_rcnn.yaml --batch_size 2 --save_to_file
```

### CMKD commands

# Then - run the scripts, and install all the missing libraries, eg:
- pip install tensorflow
- pip install opencv-python

# Run the command to prepare data
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
For the following command, the raw data needs to be organised like: `CCMKD/data/kitti/raw/KITTI_Raw/2011_10_03/2011_10_03_drive_0042_sync/image_02/data/0000000534.png`. Note that this will have to be changed when running the train.py files to: `kitti/raw/2011_10_03/2011_10_03_drive_0042_sync/image_02/data/0000000534.png` (without the KITTI_Raw folder)
```
python -m pcdet.datasets.kitti.kitti_dataset_cmkd create_kitti_infos_unlabel tools/cfgs/dataset_configs/kitti_dataset.yaml
```

# Before running training scripts, you will need to install the following libraries:
- pip install numpy==1.23.0
- pip install python-gflags
- pip install ordered-set
- pip install kornia
##  2 - Running commands

Needed to modify:

CKMD:
- First training student from scratch, trained LiDAR teacher.
```
python train_cmkd.py --cfg ../tools/cfgs/kitti_models/CMKD/CMKD-scd/cmkd_kitti_eigen_R50_scd_bev.yaml --tcp_port 16677 --pretrained_lidar_model ../ckpts/scd-teacher-kitti.pth
```

OM3D:
- First training student from scratch, trained LiDAR teacher.
```
python train_odm3d.py --cfg_file cfgs/kitti_models/ODM3D/odm3d_s1.yaml --pretrained_lidar_model ../checkpoints/scd-teacher-kitti.pth
```
- Second training:
```
```
## 3 - Visualize tensorboard results
I am using another conda env ( CMKDK) for this, to make sure the new installs dont messup the training env
```
tensorboard --logdir_spec=path_to_the_log_directory
```
or
```
tensorboard --logdir=path_to_the_log_directory
```
## Trouble-shooting:

### 1ST
When running
```
python -m pcdet.datasets.kitti.kitti_dataset_cmkd create_kitti_infos_unlabel tools/cfgs/dataset_configs/kitti_dataset.yaml
```

If you get into the error:
- " dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
TypeError: load() missing 1 required positional argument: 'Loader'" , it is because of the `pyyaml`package that doesnt have the right version. You should change it to :
```
pip install pyyaml==5.4.1
```

### 2nd
If you get the error:
```
File "/home/ubuntu/anaconda3/envs/openpcdet/lib/python3.8/site-packages/kornia/geometry/conversions.py", line 556

    # this slightly awkward construction of the output shape is to satisfy torchscript
    output_shape = [*list(quaternion.shape[:-1]), 3, 3]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    matrix = matrix_flat.reshape(output_shape)
'quaternion_to_rotation_matrix' is being compiled since it was called from 'quat_to_mat'
  File "/home/ubuntu/alex/OpenPCDet/pcdet/datasets/argo2/argo2_utils/so3.py", line 19
        (...,3,3) 3D rotation matrices.
    """
    return C.quaternion_to_rotation_matrix(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        quat_wxyz, order=C.QuaternionCoeffOrder.WXYZ
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    )
```
it is probably because of the kornia version. Try using
:
```
pip uninstall kornia
pip install kornia==0.6.2
```
### 3d
When running the demo.py file:

```
python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt ../checkpoint/kitti/pointpillar_7728.pth \
    --data_path ../data/kitti/training/velodyne
```
I run into the error:
```
vis.get_render_option().point_size = 1.0
AttributeError: 'NoneType' object has no attribute 'point_size'
```
Couldnt figure out how to make it work
It is because there is no display attached, due to : "[Open3D WARNING] GLFW Error: X11: The DISPLAY environment variable is missing
[Open3D WARNING] Failed to initialize GLFW".