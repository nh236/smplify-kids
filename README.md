# SMPLify-KiDS: Tracking <ins>K</ins>ids <ins>i</ins>n <ins>D</ins>epth <ins>S</ins>equences

<p float="center">
  <img src="assets/teaser.gif" width="50%" />
</p>

This repository contains code for fitting a human body model to RGB-D data of humans of all sizes, which was used in our paper:  
N. Hesse, S. Baumgartner, A. Gut, H. J. A. van Hedel, "*Concurrent Validity of a Custom Method for Markerless 3D Full-Body Motion Tracking of Children and Young Adults based on a Single RGB-D Camera*", 
which can be found [here](https://doi.org/10.1109/TNSRE.2023.3251440).


The code in this repository is based on [SMPLify-X](https://github.com/vchoutas/smplify-x) and we would like to thank the authors for their amazing work!

## Updates
* January 2024: added zip file containing sample data. After unpacking, run the code like this: `python smplifyx/run_rgbd_fit.py --data_folder smplifykids_sample --output_folder
                       smplifykids_sample/output --gender male --rotation 90`
* April 2023: added repo for unpacking/preprocessing mkv recordings 

## Table of Contents
  * [Installation](#installation)
  * [License](#license)
  * [Citation](#citation)
  * [Contact](#contact)

## Installation
This code was tested with Python3.10, pytorch0.7.2 and pytorch 1.12.1+cu113


#### Create virtual environment
```
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Install matching versions of pytorch3d and torch  
A list of prebuilt versions of pytorch3d can be found here: https://github.com/facebookresearch/pytorch3d/issues/1434#issue-1565285223
First select the pytorch3d version (that matches your Python and CUDA versions) and then install the corresponding torch version (that matches your CUDA version)

pytorch (https://pytorch.org/get-started/previous-versions/)
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

pytorch3d (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
# dependencies
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1121/download.html
```

#### Additional dependencies

`pip install trimesh`

Follow instructions on https://github.com/MPI-IS/mesh to install the mesh package

Install the model package: `pip install smplx[all]`

Self-intersection package (optional - if you don't want to use it set `interpenetration: False` in config file)

Follow instructions on https://github.com/vchoutas/torch-mesh-isect#installation
(you might need to copy `double_vec_ops.h` from `include` folder to `src` folder)

for preprocessing:
`pip install scikit-learn`

### Files

Download SMPL-H from http://mano.is.tue.mpg.de/ and place it in files/models

Download GMM prior from http://smplify.is.tue.mpg.de and place it in files/prior (see comment here: https://github.com/vchoutas/smplify-x/issues/51#issuecomment-544128577)

Download smpl_segmentation.pkl from https://github.com/vchoutas/torch-mesh-isect#examples ("The file for the part_segm_fn argument for SMPL can be downloaded here.") and place it in files/models

The final directory tree should look like this:
```
smplify-kids
├── cfg_files
│   ├── fit_child_rgbd_smplh.yaml
├── files
│   ├── model_templates
│   │   ├── smil_template_fingers_fix.ply
│   ├── models
│   │   ├── smplh
│   │   │   ├── SMPLH_FEMALE.pkl
│   │   │   ├── SMPLH_MALE.pkl
│   │   ├── smpl_segmentation.pkl
│   ├── prior
│   │   ├── gmm_08.pkl
│   ├── face_lm_vert_inds_SMPL_full_op_ordered.txt
│   ├── smplh_footsoles.txt
│   ├── smplh_vertex_weights_wrt_face_area.npy
├── smplifyx
│   ├── ...
├── .gitignore
├── LICENSE
└── README.md
```
### Data prepocessing
1. record mkv (e.g., using official recorder: https://learn.microsoft.com/en-us/azure/kinect-dk/azure-kinect-recorder)
2. unpack, register (and downscale) RGB and depth images using https://github.com/nh236/k4a_extract_data
3. estimate keypoints (e.g. Openpose: https://github.com/CMU-Perceptual-Computing-Lab/openpose).
   If other method is used, make sure the keypoints are transformed to Openpose format.
4. segment person (with estimated ground plane): run `python smplifyx/preprocess_data_k4a.py [-h] --data_folder DATA_FOLDER [--visualize]
                              [--rotation {0,90,180,270}]`
Expected folder structure to run fitting:
  ```
  data_folder
  ├── downscaled
  │   ├── cropped
  │   │   ├── depth
  │   │   ├── openpose
  │   │   ├── rgb
  │   │   ├── bg_plane.gre
  │   │   ├── calib.txt
  │   ├── depth
  │   ├── rgb
  │   ├── calib.txt
  ├── openpose
  ├── calib.txt
  ├── recording.mkv
```

5. run fitting
``` 
python smplifyx/run_rgbd_fit.py [-h] --data_folder DATA_FOLDER --output_folder
                       OUTPUT_FOLDER [--gender {female,male,neutral}]
                       [--visualize] [--saveonly] [--rotation {0,90,180,270}]

* data_folder: top folder containing all the data (see above)
* output_folder: results are saved here
* gender: which version of the model to use
* visualize: display fitting (makes processing a bit slower)
* saveonly: stores joint positions into separate file (if results exist for complete sequence)
* rotation: if camera was rotated during recording, but sequence is processed so that the person is upright
```


## Notes
The code relies on the following assumptions. If you do not get expected results, please check if your data conforms to these assumptions.
* We assume an Azure Kinect Developer Kit camera, but the code can be adapted to work with other cameras.
* We recorded in NFOV unbinned mode, with color resolution of 1920x1080 and depth resolution of 640x576, using the official Azure Kinect recorder (https://learn.microsoft.com/en-us/azure/kinect-dk/azure-kinect-recorder).
* Only one person should be in camera view. If there is more than one person, make sure that the first set of keypoints in each file always belongs to the person of interest.
* We assume a "clean" scene, with a person standing on a flat surface, so that it can be easily segmented from the scene. For more cluttered scenes, applying semantic segmentation algorithms can help, but we do not provide code for this.
* We do not expect the person to take a specific initialization pose, but all body parts should be visible in the first frame.

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/nh236/smplify-kids/blob/main/LICENSE) and any accompanying documentation before you download and/or use the data and software. By downloading and/or using the software & data (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the software & data. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Citation

If you find this repository useful in your research, please cite:
```
@article{hesse2023smplifykids,
  author={Hesse, Nikolas and Baumgartner, Sandra and Gut, Anja and Van Hedel, Hubertus J. A.},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  title={Concurrent Validity of a Custom Method for Markerless 3D Full-Body Motion Tracking of Children and Young Adults based on a Single RGB-D Camera},
  year={2023},
  volume={},
  number={},
  pages={},
  doi={10.1109/TNSRE.2023.3251440}}
```

This work is based on the [SMPLify-X](https://github.com/vchoutas/smplify-x) paper/repository:
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```


## Contact
The modifications in this repository (with respect to SMPLify-X) were implemented by [Nikolas Hesse](nikolas.hesse@kispi.uzh.ch), at [Swiss Children's Rehab](https://www.kispi.uzh.ch/kinder-reha), University Children's Hospital Zurich, Switzerland.