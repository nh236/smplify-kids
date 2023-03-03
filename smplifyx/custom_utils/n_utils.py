# Copyright©2023 University Children’s Hospital Zurich – Eleonore Foundation
# By using this code you agree to the terms specified in the LICENSE file

import numpy as np
from os.path import exists, join
import torch
import re
from psbody.mesh import Mesh
import smplx
from enum import Enum

sensor_Kinect4Azure = 'Kinect4Azure'

numbers = re.compile(r'(\d+)')

class RecType(Enum):
    DEFAULT = 1


def load_model(model_params):
    body_model = smplx.create(**model_params)

    return body_model

def get_vertex_weights_wrt_face_area(model_type, device, dtype):
    from paths import model_data_folder

    fn = join(model_data_folder, '{}_vertex_weights_wrt_face_area.npy'.format(model_type))

    if exists(fn):
        return torch.tensor(np.load(fn),
                            device=device,
                            dtype=dtype)
    else:
        print('vertex_weights_wrt_face_area file not found: {}! Not using those weights for m2s!'.format(fn))

    return None


def get_foot_sole_v_ids(model_type):
    from paths import model_data_folder
    fn = join(model_data_folder, '{}_footsoles.txt'.format(model_type))

    if exists(fn):
        return np.loadtxt(fn, delimiter=',', dtype=int)
    else:
        print('No foot sole vertex ids found for model {}'.format(model_type))


def read_plane_parms_from_file(filename):
    plane_parms = np.zeros(4)

    with open(filename) as f:
        for line in f.readlines():
            contents = line.split()
            if contents[0] == "BGPLANE" and len(contents) == 5:
                plane_parms = np.array(contents[1:]).astype(np.float32)

                f.close()
                break

    if (plane_parms == np.zeros(4)).all():
        print('Error reading plane parameters from file!')

    return plane_parms


def generate_seq_info_from_data(outfolder, sensor_name, rot_deg, data_folder, save_only=False):
    conf = {'paths': {'output_folder': outfolder,
                      'data_folder': data_folder
                      },
            'settings': {'do_save': True,
                         'end_frame_id': None,
                         'start_frame_id': 0,
                         'do_skip_existing': True,
                         'used_joints': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]',
                         'visualize': False,
                         'do_fit_poseonly': True,
                         'only_save_joints': save_only,
                         'do_init': True
                         },
            'initialization': {'init_start_frame_id': 0,
                               'init_end_frame_id': 5
                               },
            'scan_infos': {'use_back_view': False,
                           'rotation_deg': 0 if rot_deg is None else rot_deg,
                           'sensor_name': sensor_name
                           },
            'weights': {'init_w_betas': 10.0,
                        'w_s2m': 800.0,
                        'w_h2dlm': 0.1,  # miccai: 1., #0.1,
                        'w_f2dlm': .3,  # 1., #0.1,
                        'init_w_s2m': 100.0,
                        'init_w_poseprior': 0.1,
                        'w_poseprior': 0.15,
                        'w_mpp': 10000.0,
                        'init_w_m2s': 100.0,
                        'w_pose_smooth': 1.0,
                        'w_m2s': 400.0,
                        'w_facelm': 100.0,
                        'w_betas': 1.0,  # 5.0, #1.0 if fold_num is None else 10.0
                        'w_j2d': 0.05  # 0.1
                        },
            }

    return conf


def get_seq_info(data_folder, out_folder, rotation=0, save_only=False):

    conf = generate_seq_info_from_data(outfolder=out_folder,
                                       sensor_name=sensor_Kinect4Azure,
                                       rot_deg=rotation,
                                       data_folder=data_folder,
                                       save_only=save_only)

    return conf


def get_file_list(folder):
    from os.path import join
    import glob

    if exists(join(folder, 'depth')):
        filelist = [join(join(folder, 'depth'), f) for f in
                    sorted(glob.glob(join(join(folder, 'depth'), '*.png')), key=numerical_sort)]
    else:
        print('ERROR (get_file_list): NO DUMP FILE OR DEPTH FOLDER!!')
        return []

    return filelist


op_body_25_parent_connections = {1: 0, # head-neck
                                 2:1, 3:2, 4:3, # right arm
                                 5: 1, 6:5, 7:6, # left arm
                                 8:1, # hip
                                 9:8, 10:9, 11:10, 24:11, 22:11, 23:22, # right leg
                                 12:8, 13:12, 14:13, 21:14, 19:14, 20:19, # left leg
                                 15:0, 17:15, 16:0, 18:16} # eyes, ears


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# signed dist to plane
def get_signed_dists_to_plane(vertices, plane_params, buffer=-0.005, device=None, do_flip=True):
    #plane_params = torch.tensor(plane_params, device=vertices.device)
    plane_normal = plane_params[0:3]
    point_on_plane = plane_normal * -plane_params[3]

    if do_flip:
        signfac = -1 if plane_normal[1] >= 0 else 1
    else:
        signfac = 1 if plane_normal[1] >= 0 else -1

    result = signfac * torch.matmul(plane_normal, (point_on_plane - vertices).T)

    return result + buffer


##############################
# Cameras
##############################
def load_k4a_calibration(calib_filename):
    import cv2
    fs = cv2.FileStorage(calib_filename, cv2.FILE_STORAGE_READ)
    fn = fs.getNode("color_cam_matrix")
    cam_mat = fn.mat()

    fn = fs.getNode("color_dist_coeffs")
    dist_coeffs = fn.mat()

    return cam_mat, dist_coeffs


def get_depth_cam_torch(sensor_name,
                        portrait_mode,
                        dtype=torch.float32,
                        k4a_filename=None):
    from camera import create_camera

    if sensor_name == sensor_Kinect4Azure:
        if not exists (k4a_filename):
            print('ERROR: No Kinect4Azure calibration file!!!')
            return None

        # xy switched?!?
        cam_mat, dist_coeffs = load_k4a_calibration(k4a_filename)
        fy = cam_mat[0, 0]
        fx = cam_mat[1, 1]
        cx = cam_mat[0, 2]
        cy = cam_mat[1, 2]

        if portrait_mode:
            cy = cam_mat[0, 2]
            cx = cam_mat[1, 2]
        # images already undistorted!!

    c = torch.tensor([cx, cy]).unsqueeze(0)

    cam = create_camera(focal_length_x=torch.tensor([fx]),
                        focal_length_y=torch.tensor([fy]),
                        center=c,
                        dtype=dtype)

    return cam


def get_verts_from_depth_im_torch(depth_im, camera, sensor=None):
    indices = np.where((depth_im < 65000.) & (depth_im > 0.))

    verts = np.zeros((indices[0].size, 3))
    verts[:,2] = depth_im[indices].ravel() / 1000.

    verts[:, 0] = (np.asarray(indices).T[:, 1] - camera.center[0, 0].detach().cpu().numpy()) * verts[:, 2] / \
                  camera.focal_length_x[0].detach().cpu().numpy()
    verts[:, 1] = (np.asarray(indices).T[:, 0] - camera.center[0, 1].detach().cpu().numpy()) * verts[:, 2] / \
                  camera.focal_length_y[0].detach().cpu().numpy()

    return verts, indices


def load_depth_and_rgb_file(scan_folder, filename, load_rgb=True):
    import cv2

    depth_im = None # np.array([])
    rgb_im = None # np.array([])

    if exists(join(scan_folder, 'depth/' + filename)):
        depth_im = cv2.imread(join(scan_folder, 'depth/' + filename), -1)

    if load_rgb and exists(join(scan_folder, 'rgb/' + filename[:filename.rfind('_depth')] + '.png')):
        rgb_im = cv2.imread(join(scan_folder, 'rgb/' + filename[:filename.rfind('_depth')] + '.png'))[:, :, ::-1] / 255.
    # import matplotlib.pyplot as plt
    # plt.imshow(rgb_im)

    return depth_im, rgb_im


def load_frame_torch(scan_folder, cloud_filename, conf_scan_infos, dtype, load_rgb=True):
    depth_im, rgb_im = load_depth_and_rgb_file(scan_folder, cloud_filename, load_rgb=load_rgb)

    k4a_calib_fn = None

    if conf_scan_infos['sensor_name'] == sensor_Kinect4Azure:
        k4a_calib_fn = join(scan_folder, "calib.txt")

    isinportraitmode = conf_scan_infos['rotation_deg'] == 90 or conf_scan_infos['rotation_deg'] == 270

    # Only depth cam is torch, rest is numpy arrays
    depth_cam = get_depth_cam_torch(conf_scan_infos['sensor_name'], isinportraitmode,
                                    dtype=dtype, k4a_filename=k4a_calib_fn)

    pc_v, pc_v_inds = get_verts_from_depth_im_torch(depth_im, depth_cam, conf_scan_infos['sensor_name'])

    return depth_im, rgb_im, pc_v, pc_v_inds


def load_poses_from_json(filename, smooth_addition=''):
    import json

    with open(filename, 'r') as jsonfile:
        keypoint_dict = json.load(jsonfile)

        if len(keypoint_dict['people']) > 0:
            if len(keypoint_dict['people']) > 1:
                print('{} people detected!'.format(len(keypoint_dict['people'])))

            kps = keypoint_dict['people'][0]
            op_face_lms_2D_conf = np.asarray(kps['face_keypoints_2d']).reshape(-1, 3)
            op_pose_lms_2D_conf = np.asarray(kps['pose_keypoints_2d']).reshape(-1, 3)
            op_hand_left_lms_2D_conf = np.asarray(kps['hand_left_keypoints_2d']).reshape(-1, 3)
            op_hand_right_lms_2D_conf = np.asarray(kps['hand_right_keypoints_2d']).reshape(-1, 3)

        jsonfile.close()

    return op_face_lms_2D_conf, op_pose_lms_2D_conf, op_hand_left_lms_2D_conf, op_hand_right_lms_2D_conf


def write_poses_to_json(filename, body_kps, face_kps, lhand_kps, rhand_kps):
    import json

    keypoint_dict = {'version': 1.3,
                     'people': [{'person_id': [-1],
                                 'face_keypoints_2d': face_kps.flatten().tolist(),
                                 'pose_keypoints_2d': body_kps.flatten().tolist(),
                                 'hand_left_keypoints_2d': lhand_kps.flatten().tolist(),
                                 'hand_right_keypoints_2d': rhand_kps.flatten().tolist(),
                                 'pose_keypoints_3d': [],
                                 'face_keypoints_3d': [],
                                 'hand_left_keypoints_3d': [],
                                 'hand_right_keypoints_3d': []}]
                     }
    with open(filename, 'w') as jsonfile:
        json.dump(keypoint_dict, jsonfile)


def load_poses_from_files(scan_folder, cloud_filename, smooth_addition=''):
    # if pose yml file exists:
    if exists(join(scan_folder,
                   'openpose/' + cloud_filename[
                                 :cloud_filename.rfind('_depth')] + '_keypoints' + smooth_addition + '.json')):
        op_face_lms_2D_conf, op_pose_lms_2D_conf, op_hand_left_lms_2D_conf, op_hand_right_lms_2D_conf = load_poses_from_json(
            join(scan_folder,
                 'openpose/' + cloud_filename[
                               :cloud_filename.rfind('_depth')] + '_keypoints' + smooth_addition + '.json'),
            smooth_addition)
    else:
        print('No keypoint files found for file: %s' % cloud_filename)

        return None, None, None, None

    return op_face_lms_2D_conf, op_pose_lms_2D_conf, op_hand_left_lms_2D_conf, op_hand_right_lms_2D_conf


############################
# rendering #
############################
def combine_meshes(meshes, color=np.array([0.7, 0.7, 0.9])):
    v = []
    vc = []
    f = []
    N = 0
    for m in meshes:
        v.append(m.v)
        if hasattr(m, 'vc'):
            vc.append(m.vc)
        else:

            vc.append(np.tile(color, (m.v.shape[0], 1)))
        f.append(m.f + N)
        N += m.v.shape[0]

    v = np.concatenate(v)
    vc = np.concatenate(vc)
    f = np.concatenate(f)
    return Mesh(v=v, vc =vc, f=f)


def get_joint_weights(num_body_joints,
                      use_hands=False,
                      use_face=False,
                      use_face_contour=False,
                      dtype=torch.float32):
        # The weights for the joint terms in the optimization
        optim_weights = torch.ones(num_body_joints + 2 * use_hands +
                                use_face * 51 +
                                17 * use_face_contour,
                                dtype=np.float32)

        return optim_weights


def get_v_inds_for_parts(bm, part_index_list):
    part_argmax = np.argmax(bm.lbs_weights.detach().cpu().numpy(), axis=1)

    part_filter = np.any(np.array([np.any([part_argmax == p_i], axis=0) for p_i in part_index_list]), axis=0)
    v_inds = np.where(part_filter)

    return np.asarray(v_inds).squeeze()


def get_foot_v_ids(bm):

    part_argmax = np.argmax(bm.lbs_weights.detach().cpu().numpy(), axis=1)
    part_filter_lfoot = np.any([part_argmax == 7, part_argmax == 10], axis=0)
    lfoot_v_ids = np.where(part_filter_lfoot)
    part_filter_rfoot = np.any([part_argmax == 8, part_argmax == 11], axis=0)
    rfoot_v_ids = np.where(part_filter_rfoot)

    return lfoot_v_ids, rfoot_v_ids


def py3d_point_mesh_face_distance_w_inds(meshes, pcls):
    # taken from pytorch3d.loss.point_mesh_distance.point_mesh_face_distance

    """
        Computes the distance between a pointcloud and a mesh within a batch.
        Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
        sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

        `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
            to the closest triangular face in mesh and averages across all points in pcl
        `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
            mesh to the closest point in pcl and averages across all faces in mesh.

        The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
        and then averaged across the batch.

        Args:
            meshes: A Meshes data structure containing N meshes
            pcls: A Pointclouds data structure containing N pointclouds

        Returns:
            loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
                between all `(mesh, pcl)` in a batch averaged across the batch.
        """
    # from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance
    from .point_mesh_distance import point_face_distance, face_point_distance

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    use_weighting = False

    # !!!!!!!!!! in  pytorch3d/pytorch3d/csrc/utils/geometry_utils.h
    # in IsInsideTriangle
    # in BarycentricCoords3Forward:
    # line 596:  const T denom = d00 * d11 - d01 * d01 + kEpsilon;
    # if d's become very small, kEpsilon is too big!! 1e-8
    # easy hack: multiply by 100 before passing to point_face_distance and face_point_distance (and divide by 100**2 in the end)
    # https://github.com/facebookresearch/pytorch3d/blob/cc70950f4064e3feeb55281b829aa55aa4a7e942/pytorch3d/csrc/utils/geometry_utils.h

    # packed representation for pointclouds
    points = pcls.points_packed() * 100.  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed() * 100.
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, p2f_inds = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    if use_weighting:
        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = point_to_face * weights_p
        point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point, f2p_inds = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris
    )

    if use_weighting:
        # weight each example by the inverse of number of faces in the example
        tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
        num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
        weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
        weights_t = 1.0 / weights_t.float()
        face_to_point = face_to_point * weights_t
        face_dist = face_to_point.sum() / N

    # 1e4 = 100 ** 2 (see points and verts scaling above)
    return (point_to_face / 1e4).reshape(1, -1), face_to_point.reshape(1, -1) / 1e4, p2f_inds, f2p_inds
