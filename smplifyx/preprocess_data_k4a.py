# Copyright©2023 University Children’s Hospital Zurich – Eleonore Foundation
# By using this code you agree to the terms specified in the LICENSE file

import torch
import numpy as np
import custom_utils.n_utils as n_utils
from os.path import join, exists
from itertools import chain


def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix.

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal


def local_regression_plane_ransac(points, plane_thresh):
    from sklearn import linear_model
    """
    Computes parameters for a local regression plane using RANSAC
    """

    XY = np.concatenate((points[:,0, np.newaxis], points[:,2, np.newaxis]), axis=1) #points[:,:2]
    Z = points[:,1] # points[:,2]
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                          residual_threshold=plane_thresh
                                          )
    ransac.fit(XY, Z)

    inlier_mask = ransac.inlier_mask_

    a,b,c,d = best_fitting_plane(points[inlier_mask], True)

    return a,b,c,d,inlier_mask


#################################################
# identify what's baby and what's background based on table plane
#################################################
def get_dist_to_plane(points, plane_parms):
    plane_normal = np.array(plane_parms[0:3])
    point_on_plane = plane_normal * -plane_parms[3]

    # if do_flip:
    #     signfac = -1 if plane_normal[2] >= 0 else 1
    # else:
    #     signfac = 1 if plane_normal[2] >= 0 else -1

    signfac = 1.
    result = signfac * np.dot(plane_normal,(point_on_plane-points).T)

    return result


#################################################
# rotate and segment files
#################################################
def segment_data(data_folder, do_display=False, do_downscale=False, rotation_deg=0):
    from os import makedirs
    from os.path import basename, dirname, normpath
    import shutil
    import yaml

    from psbody.mesh import Mesh
    import cv2

    ### options #############
    refine_with_pose = True

    do_save = True
    skip_existing = False

    dataset_weight_conf_fn = join('cfg_files', 'fit_child_rgbd_smplh.yaml')

    if exists(dataset_weight_conf_fn):
        with open(dataset_weight_conf_fn, 'r') as conf_file:
            all_settings = yaml.safe_load(conf_file)

    plane_thresh = 0.015
    distance_thresh = 7000
    bg_depth_thresh = 10
    do_segment = True

    if not do_segment:
        print('NO SEGMENTATION APPLIED!!!')

    sensor_name = n_utils.sensor_Kinect4Azure
    raw_seq_folder = data_folder
    keypoint_base_folder = raw_seq_folder

    if do_downscale:
        raw_seq_folder = join(raw_seq_folder, 'downscaled')
        # openpose keypoints not downscaled yet!

    out_folder = join(raw_seq_folder, 'cropped')
    plane_folder = out_folder
    out_depth = join(out_folder, 'depth')
    out_rgb = join(out_folder, 'rgb')
    out_keypoints_folder = join(out_folder, 'openpose')

    if not exists(out_folder):
        makedirs(out_folder)
    if not exists(plane_folder):
        makedirs(plane_folder)
    if not exists(out_depth):
        makedirs(out_depth)
    if not exists(out_rgb):
        makedirs(out_rgb)
    if not exists(out_keypoints_folder):
        makedirs(out_keypoints_folder)

    print(out_folder)

    k4a_calib_fn = None

    # load config
    conf = n_utils.get_seq_info(data_folder=data_folder,
                                out_folder=out_folder,
                                rotation=rotation_deg)

    if conf['scan_infos']['sensor_name'] == n_utils.sensor_Kinect4Azure:
        k4a_calib_fn = join(raw_seq_folder, "calib.txt")

        if exists(k4a_calib_fn):
            shutil.copy(k4a_calib_fn, out_folder)

    isinportraitmode = rotation_deg == 90 or rotation_deg == 270
    filelist = n_utils.get_file_list(raw_seq_folder)

    depth_cam = n_utils.get_depth_cam_torch(sensor_name=sensor_name,
                                            portrait_mode=isinportraitmode,
                                            k4a_filename=k4a_calib_fn)
    plane_parms = None

    # Downscale ALL keypoints first
    # get all keypoints with confidence > threshold
    # create mask image
    # dilate
    # use as fg mask

    good_init_frame = 0
    frame_list = [x for x in
                  chain(range(good_init_frame, len(filelist)), range(0, good_init_frame + 1)[::-1])]

    for frame_num in frame_list:
        file = filelist[frame_num]
        cloud_filename = basename(file)

        kp_out_filename = join(out_keypoints_folder,
                               cloud_filename[:cloud_filename.rfind('_depth')] + '_keypoints.json')

        if skip_existing and exists(kp_out_filename):
            face_kps, body_kps, lhand_kps, rhand_kps = \
                n_utils.load_poses_from_files(out_folder, cloud_filename, smooth_addition='')

            if face_kps is None and body_kps is None and lhand_kps is None and rhand_kps is None:
                continue
        else:
            # load openpose joints
            face_kps, body_kps, lhand_kps, rhand_kps = \
                n_utils.load_poses_from_files(keypoint_base_folder, cloud_filename, smooth_addition='')

            if face_kps is None and body_kps is None and lhand_kps is None and rhand_kps is None:
                continue

            # add dimension to have same format ("batch dimension"?)
            body_kps = np.expand_dims(body_kps, axis=0)
            face_kps = np.expand_dims(face_kps, axis=0)
            lhand_kps = np.expand_dims(lhand_kps, axis=0)
            rhand_kps = np.expand_dims(rhand_kps, axis=0)

            # DOWNSCALE!!!
            if do_downscale:
                print('WARNING: DOWNSCALING OPENPOSE KEYPOINTS HERE!!')
                if face_kps is not None:
                    face_kps[:, :, :2] = face_kps[:, :, :2] / 2.
                if body_kps is not None:
                    body_kps[:, :, :2] = body_kps[:, :, :2] / 2.
                lhand_kps[:, :, :2] = lhand_kps[:, :, :2] / 2.
                rhand_kps[:, :, :2] = rhand_kps[:, :, :2] / 2.

                if not exists(kp_out_filename):
                    n_utils.write_poses_to_json(kp_out_filename, body_kps=body_kps, face_kps=face_kps,
                                                 lhand_kps=lhand_kps, rhand_kps=rhand_kps)

    keypoint_base_folder = out_folder

    last_valid_kps = {}

    # Segment files
    for frame_num in frame_list:
        file = filelist[frame_num]
        cloud_filename = basename(file)

        if skip_existing and \
            exists(join(out_rgb, cloud_filename[:cloud_filename.rfind('_depth')] + '.png')) and \
            exists(join(out_depth, cloud_filename[:cloud_filename.rfind('.png')] + '_new.png')):
            continue

        ### Load frame ###
        depth_im, rgb_im, pc_v, pc_v_inds = n_utils.load_frame_torch(raw_seq_folder, cloud_filename, conf['scan_infos'],
                                                                 dtype=torch.float32, load_rgb=True)

        # distance thresholding
        if distance_thresh > 0:
            depth_im_mask = depth_im > distance_thresh

        if not isinportraitmode:
            depth_im_mask[:, :50] = True
            depth_im_mask[:, -50:] = True

            depth_im[depth_im_mask] = 65000

        rot_depth_im = depth_im
        rot_rgb_im = rgb_im[:, :,::-1]

        # don't throw away missing pixels!
        miss_pix_mask = (rot_depth_im == 0)

        ### bg plane segment ###
        if do_segment:
            if plane_parms is None:
                # only save bg_plane once
                if exists(join(plane_folder, 'bg_plane.gre')):
                    plane_parms = n_utils.read_plane_parms_from_file(join(plane_folder, 'bg_plane.gre'))
                elif exists(join(raw_seq_folder, 'bg_plane.gre')):
                    plane_parms = n_utils.read_plane_parms_from_file(join(raw_seq_folder, 'bg_plane.gre'))

                    with open(join(plane_folder, 'bg_plane.gre'), 'w') as plane_file:
                        plane_file.write(
                         'BGPLANE %f %f %f %f' % (plane_parms[0], plane_parms[1], plane_parms[2], plane_parms[3]))

                else:
                    # inliers_mask: True for point belonging to floor
                    plane_points_limit_X = 2.
                    plane_points_limit_Y = 1.
                    plane_points_limit_Z = 4.
                    bg_plane_pc_mask = (pc_v[:, 0] < plane_points_limit_X) * \
                                       (pc_v[:, 0] > -plane_points_limit_X) * \
                                       (pc_v[:, 1] > -plane_points_limit_Y) * \
                                       (pc_v[:, 2] < plane_points_limit_Z)

                    if do_display:
                        Mesh(v=pc_v[bg_plane_pc_mask], f=[]).show()

                    a,b,c,d, inlier_mask = local_regression_plane_ransac(pc_v[bg_plane_pc_mask], plane_thresh)

                    plane_parms = np.array([a, b, c, d]) * -1.

                    if plane_parms[1] < 0.:
                        plane_parms *= -1.

                    if do_display:
                        plane_vertices = []
                        plane_size = 1.


                        plane_normal = np.array(plane_parms[0:3])
                        x = np.array([1., 1., 1.])
                        x -= x.dot(plane_normal) * plane_normal / np.linalg.norm(plane_normal) ** 2
                        y = np.cross(plane_normal, x)

                        point_on_plane = plane_normal * -plane_parms[3]

                        plane_vertices.append(point_on_plane + (plane_size * x))
                        plane_vertices.append(point_on_plane - (plane_size * x))
                        plane_vertices.append(point_on_plane - (plane_size * y))
                        plane_vertices.append(point_on_plane + (plane_size * y))

                        plane_vertices = np.asarray(plane_vertices)

                        plane_faces = np.array([[0, 1, 2], [0, 1, 3]])

                        from psbody.mesh import MeshViewer
                        mv = MeshViewer('test')
                        plane_mesh = Mesh(v=plane_vertices, f=plane_faces)
                        mv.set_static_meshes([Mesh(v=pc_v, f=[]), plane_mesh])

                    plane_out_fn = join(plane_folder, 'bg_plane.gre')

                    with open(plane_out_fn, 'w') as plane_file:
                        plane_file.write(
                         'BGPLANE %f %f %f %f' % (plane_parms[0], plane_parms[1], plane_parms[2], plane_parms[3]))


            dists_to_bg_plane_pc_v = get_dist_to_plane(pc_v, plane_parms)

            # bg_mask_pc_v = np.ones_like(dists_to_bg_plane_pc_v).astype(bool)
            bg_mask_pc_v = (dists_to_bg_plane_pc_v < plane_thresh)

            dists_to_bg_plane_im = np.zeros(rot_depth_im.shape, np.float64)
            dists_to_bg_plane_im[pc_v_inds[0], pc_v_inds[1]] = dists_to_bg_plane_pc_v

            seg_rot_depth_im = np.copy(rot_depth_im)
            seg_rot_depth_im[pc_v_inds[0][bg_mask_pc_v], pc_v_inds[1][bg_mask_pc_v]] = 65000
            bg_mask_im = (seg_rot_depth_im == 65000) + (seg_rot_depth_im == 0)

            if exists(join(raw_seq_folder, 'bg_depth.png')):
                bg_depth_im = cv2.imread(join(raw_seq_folder, 'bg_depth.png'), -1)
                bg_depth_valid_mask = (bg_depth_im < 64000) * (bg_depth_im > 0.)
                depth_diff_mask = np.abs(seg_rot_depth_im.astype(np.float32) - bg_depth_im.astype(np.float32)) < bg_depth_thresh


                bg_mask_im[depth_diff_mask * bg_depth_valid_mask] = True

            # load openpose joints
            face_kps, body_kps, lhand_kps, rhand_kps = \
                n_utils.load_poses_from_files(keypoint_base_folder, cloud_filename, smooth_addition='')

            if refine_with_pose and body_kps is not None:
                ### segment foreground using openpose joints ###
                # add dimension to have same format ("batch dimension")
                body_kps = np.expand_dims(body_kps, axis=0)
                face_kps = np.expand_dims(face_kps, axis=0)
                lhand_kps = np.expand_dims(lhand_kps, axis=0)
                rhand_kps = np.expand_dims(rhand_kps, axis=0)

                # if confidence == 0 use last valid kp
                if 'body_kps' in last_valid_kps:
                    body_kps[body_kps[:, :, 2] == 0.] = last_valid_kps['body_kps'][body_kps[:, :, 2] == 0.]
                    face_kps[face_kps[:, :, 2] == 0.] = last_valid_kps['face_kps'][face_kps[:, :, 2] == 0.]
                    lhand_kps[lhand_kps[:, :, 2] == 0.] = last_valid_kps['lhand_kps'][lhand_kps[:, :, 2] == 0.]
                    rhand_kps[rhand_kps[:, :, 2] == 0.] = last_valid_kps['rhand_kps'][rhand_kps[:, :, 2] == 0.]

                if False:
                    tmp_img = rgb_im.copy()
                    for coord in body_kps[0,:,:2]:
                        if (0 <= coord[0] < tmp_img.shape[1] and
                                0 <= coord[1] < tmp_img.shape[0]):
                            cv2.circle(tmp_img, tuple((coord + .5).astype(int)), 3, [0, 0, 1])

                    cv2.imshow('kps', tmp_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                final_seg_mask = None

                if body_kps.size > 2:

                    keypoint_mask = np.zeros_like(rot_depth_im)

                    if True:
                        all_kps = np.concatenate((body_kps, face_kps, lhand_kps, rhand_kps), axis=1)
                    else:
                        all_kps = np.concatenate((body_kps, face_kps), axis=1)

                    kp_coords = np.asarray(
                        [[int(j + .5), int(i + .5)] for kp in all_kps for i, j, c in kp
                         if c > 0.2 and 0 <= int(i + .5) < rot_rgb_im.shape[1] and 0 <= int(j + .5) <
                         rot_rgb_im.shape[0]])
                    keypoint_mask[tuple(zip(*kp_coords))] = 1
                    # Use connections between joints in addition to joint positions as keypoint mask
                    for k, v in n_utils.op_body_25_parent_connections.items():
                        if body_kps[0, k, 2] > 0.2 and body_kps[0, v, 2] > 0.2:
                            keypoint_mask = cv2.line(keypoint_mask,
                                                     tuple(body_kps[0, k, :2].astype(int)),
                                                     tuple(body_kps[0, v, :2].astype(int)), color=[1],
                                                     thickness=1)

                    keypoint_dil_mask = cv2.dilate(keypoint_mask.astype(np.uint8),
                                                   np.ones((80, 80), np.uint8)).astype(bool)
                    final_seg_mask = keypoint_dil_mask


                    neigh_thresh = 50

                    temp_1 = (np.abs((seg_rot_depth_im[1:, 1:]).astype(int) - (seg_rot_depth_im[:-1, :-1]).astype(int)) < neigh_thresh) * ~bg_mask_im[1:,1:]
                    temp_2 = (np.abs((seg_rot_depth_im[1:, 1:]).astype(int) - (seg_rot_depth_im[:-1, 1:]).astype(int)) < neigh_thresh) * ~bg_mask_im[1:,1:]
                    temp_3 = (np.abs((seg_rot_depth_im[1:, 1:]).astype(int) - (seg_rot_depth_im[1:, :-1]).astype(int)) < neigh_thresh) * ~bg_mask_im[1:,1:]

                    conn_mask = np.zeros_like(rot_depth_im)
                    conn_mask[1:, 1:] = (temp_1.astype(int) + temp_2.astype(int) + temp_3.astype(int)) == 3
                    #conn_mask = cv2.dilate(conn_mask, np.ones((3, 3), np.uint8))

                    #ret, labels = cv2.connectedComponents(conn_mask.astype(np.uint8))
                    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        conn_mask.astype(np.uint8) * 255)

                    good_kp_clusters = []
                    good_cluster_num_pts_list = []
                    kp_clusters_mask = np.zeros_like(labels).astype(bool)

                    for ci in range(1, np.max(labels) + 1):
                        cur_cluster_mask = labels == ci

                        if True in cur_cluster_mask[keypoint_mask.astype(bool)]:
                            # keypoint/"bone" in current cluster
                            good_kp_clusters.append(ci)
                            kp_clusters_mask[labels==ci] = True
                            good_cluster_num_pts_list.append(stats[ci, 4])

                    max_fg_cluster = good_kp_clusters[np.argmax(good_cluster_num_pts_list)]
                    max_depth_val_fg = np.max(seg_rot_depth_im[labels == max_fg_cluster])

                    good_clusters = []
                    for ci in range(1, np.max(labels) + 1):
                        cur_cluster_mask = labels == ci

                        if True in cur_cluster_mask[keypoint_mask.astype(bool)]:
                            # keypoint/"bone" in current cluster
                            good_clusters.append(ci)
                        else:
                            # get neighborhood of current cluster
                            cur_cluster_mask = cv2.dilate(cur_cluster_mask.astype(np.uint8),
                                                          np.ones((100, 100), np.uint8)).astype(bool)

                            # get max depth values of max_fg_cluster from this neighborhood region
                            depth_overlap = seg_rot_depth_im[cur_cluster_mask * kp_clusters_mask]

                            if depth_overlap.size > 0:
                                # cluster is good if close to/smaller than max_depth_val of (local) max_fg_cluster max depth
                                if 0 < np.median(seg_rot_depth_im[labels == ci]) < np.max(depth_overlap):
                                    good_clusters.append(ci)

                    depth_bound_mask = np.zeros_like(labels).astype(bool)

                    for ci in good_clusters:
                        depth_bound_mask[labels == ci] = True

                    final_seg_mask *= depth_bound_mask * ~(bg_mask_im)

                else:
                    print('No body pose detected!')
                    final_seg_mask = ~bg_mask_im
            else:
                final_seg_mask = ~bg_mask_im
        else:
            final_seg_mask = np.ones_like(rot_depth_im)

        final_depth_im = seg_rot_depth_im
        final_depth_im[final_seg_mask==0] = 65000
        final_depth_im[miss_pix_mask] = 0   # restore missing pixels information


        if do_save:
            ### write files
            if rot_rgb_im.dtype == np.float64:
                rot_rgb_im = (rot_rgb_im * 255).astype(np.uint8)

            cv2.imwrite(join(out_rgb, cloud_filename[:cloud_filename.rfind('_depth')] + '.png'), rot_rgb_im)
            cv2.imwrite(join(out_depth, cloud_filename[:cloud_filename.rfind('.png')] + '_new.png'), final_depth_im)

        if do_display:
            final_pc_v, final_pc_v_inds = n_utils.get_verts_from_depth_im_torch(final_depth_im,
                                                                                camera=depth_cam,
                                                                                sensor=sensor_name)
            Mesh(v=final_pc_v, f=[]).show()

            temp_seg_rgb = (rot_rgb_im * (1. * np.repeat(final_seg_mask, 3).reshape(rot_rgb_im.shape))).astype(
                np.uint8)

            cv2.imshow('rgb image', rgb_im)
            cv2.imshow('rotated rgb image', rot_rgb_im)
            cv2.imshow('temp_seg_rgb', temp_seg_rgb)
            cv2.imshow('final_depth_im', final_depth_im )
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        print(cloud_filename)

        if 'body_kps' not in last_valid_kps:
            last_valid_kps['body_kps'] = body_kps
            last_valid_kps['face_kps'] = face_kps
            last_valid_kps['lhand_kps'] = lhand_kps
            last_valid_kps['rhand_kps'] = rhand_kps
        else:
            # update kps with confidence > 0.
            last_valid_kps['body_kps'][body_kps[:, :, 2] > 0.] = body_kps[body_kps[:, :, 2] > 0.]
            last_valid_kps['face_kps'][face_kps[:, :, 2] > 0.] = face_kps[face_kps[:, :, 2] > 0.]
            last_valid_kps['lhand_kps'][lhand_kps[:, :, 2] > 0.] = lhand_kps[lhand_kps[:, :, 2] > 0.]
            last_valid_kps['rhand_kps'][rhand_kps[:, :, 2] > 0.] = rhand_kps[rhand_kps[:, :, 2] > 0.]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True, type=str, help='Top folder contating the data.')
    parser.add_argument('--visualize', required=False, action='store_true', help='Shows visualization of the fitting process.')
    parser.add_argument('--rotation', required=False, default=0, type=int, choices=[0, 90, 180, 270],
                        help='If camera is rotated set degrees of rotation here.')

    args = parser.parse_args()

    segment_data(data_folder=args.data_folder, do_display=args.visualize, do_downscale=True,
                 rotation_deg=args.rotation)
