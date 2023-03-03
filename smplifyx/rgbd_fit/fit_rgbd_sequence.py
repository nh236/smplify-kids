# Copyright©2023 University Children’s Hospital Zurich – Eleonore Foundation
# By using this code you agree to the terms specified in the LICENSE file

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import sys

import pickle
from os.path import basename, join, exists, splitext, expandvars, normpath
from os import makedirs
from itertools import chain

from trimesh import Trimesh, load_mesh

import paths
import fitting
from custom_utils import n_utils
from camera import create_camera
from prior import create_prior

from utils import smpl_to_openpose, GMoF, JointMapper

from rgbd_fit.process_one_frame import process_one_frame

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings

DO_SAVE = True
do_save_plys = False
use_cuda = True

########################################
# main loop function
########################################
def run_fitting(data_folder, output_folder, gender, cam_rotation=0, do_viz=False, do_save_joints=False):
    from mesh_viewer import MeshViewer

    from timeit import default_timer as timer
    import yaml
    from datetime import datetime

    model_folder = paths.model_folder
    model_type = 'smplh'

    dtype = torch.float32

    use_hands = True
    use_face = False
    use_face_contour = False
    use_m2s = True
    do_poseonly = False
    interpenetration = False

    joint_weight_list = None

    dataset_weight_conf_fn = join('cfg_files', 'fit_child_rgbd_smplh.yaml')

    if exists(dataset_weight_conf_fn):
        with open(dataset_weight_conf_fn, 'r') as conf_file:
            all_settings = yaml.safe_load(conf_file)
            model_type = all_settings['model_type'].lower()

            opt_weights_dict = all_settings['weights']
            use_hands = all_settings['use_hands']
            use_face = all_settings['use_face']
            use_face_contour = all_settings['use_face_contour']
            use_m2s = all_settings['use_m2s']
            if 'poseonly' in all_settings.keys():
                do_poseonly = all_settings['poseonly']
            if 'interpenetration' in all_settings.keys():
                interpenetration = all_settings['interpenetration']
                ign_part_pairs = all_settings['ign_part_pairs']
                penalize_outside = all_settings['penalize_outside']
                max_collisions = all_settings['max_collisions']
                df_cone_height = all_settings['df_cone_height']
            else:
                interpenetration = False
                ign_part_pairs = None
                penalize_outside = False
                max_collisions = 8
                df_cone_height = 0.0001

            if 'joint_weight_list' in all_settings.keys():
                # list of lists [[Joint_ID, weight], [Joint_ID, weight], ...]
                joint_weight_list = all_settings['joint_weight_list']

            rho_j2d_list = all_settings['rho_j2d_list']
            rho_s2m_list = all_settings['rho_s2m_list']
            rho_m2s_list = all_settings['rho_m2s_list']
            rho_close_plane = all_settings['rho_close_plane']
            pose_prior_type = all_settings['pose_prior_type']
            im_wid = all_settings['image_width']
            im_hgt = all_settings['image_height']
    else:
        print('No weights config file found at {}. Using default values!'.format(dataset_weight_conf_fn))

        opt_weights_dict = {  # Weights
            'data_weight': [0.5, 0.5, 0.5],
            'shape_weight': [5e0, 1e0, 1e0],
            'body_pose_weight': [1e0, 5e-1, 1e-1],
            'hand_prior_weight': [1e0, 1e0, 1e-1],
            'hand_joints_weights': [1e0, 1e0, 5e0],
            's2m_weight': [1e5, 2e5, 1e6],
            'm2s_weight': [1e5, 1e5, 1e4],
            'trans_reg_weight': [1e1, 1e2, 1e2],
            'smooth_reg_weight': [1e0, 1e0, 1e0],
            'bending_prior_weight': [1e0, 1e0, 1e-1],
            'inside_plane_weight': [1e8, 1e8, 1e8],
            'close_plane_weight': [1e10, 1e10, 5e9],
            'tolerance': [1e-3, 1e-4, 1e-5],
            'coll_loss_weight': [1e-1, 1e-1, 1.],
        }

        # rho sigmas
        rho_j2d_list = [500, 50, 10] #20
        rho_s2m_list = [3e-1, 1e-1, 3e-2] #8
        rho_m2s_list = [3e-1, 1e-1, 3e-2] #8
        rho_close_plane = 1e-2
        pose_prior_type = 'gmm'

    # Load priors
    num_pca_comps = 12
    left_hand_prior_type = 'l2'
    right_hand_prior_type = 'l2'
    point2plane = False

    left_hand_prior, right_hand_prior = None, None

    if use_hands:
        left_hand_prior = create_prior(prior_type=left_hand_prior_type, dtype=dtype, use_left_hand=True)

        right_hand_prior = create_prior(prior_type=right_hand_prior_type, dtype=dtype, use_right_hand=True)

    shape_prior = create_prior(prior_type='l2',
                               model_folder='',
                               prior_folder='',
                               dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    body_color = (1.0, 1.0, 0.9, 1.0)
    viewer_flags = {}

    mv = MeshViewer(body_color=body_color, viewer_flags=viewer_flags) if do_viz else None

    optimize_scale = True

    num_betas = 10
    global_orient = torch.zeros([1, 3], dtype=dtype)
    body_pose = None
    betas = torch.zeros([1, num_betas], dtype=torch.float32)

    global_orient[0, 0] = 3.1415

    body_pose_prior = create_prior(prior_type=pose_prior_type, num_gaussians=8,
                                   prior_folder=paths.prior_folder,
                                   dtype=dtype, model_type=model_type)

    smpl2op_mapping = smpl_to_openpose(model_type=model_type, use_hands=use_hands, use_face=use_face,
                                       use_face_contour=use_face_contour, openpose_format='coco25')  # coco25

    joint_mapper = None

    if not do_save_joints:
        joint_mapper = JointMapper(smpl2op_mapping)

    create_member_vars = True

    model_params = dict(model_path=model_folder,
                        model_type=model_type,
                        joint_mapper=joint_mapper,
                        create_glb_pose=True,
                        use_hands=use_hands,
                        body_pose=body_pose,
                        global_orient=global_orient,
                        create_body_pose=create_member_vars,
                        create_betas=create_member_vars,
                        betas=betas.clone(),
                        create_left_hand_pose=create_member_vars,
                        create_right_hand_pose=create_member_vars,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=False,
                        create_reye_pose=False,
                        create_transl=create_member_vars,
                        dtype=dtype,
                        use_compressed=False,
                        num_betas=num_betas,
                        gender=gender
                        )


    body_model = n_utils.load_model(model_params=model_params)

    if use_face and model_type == 'smplh':
        from smplx.vertex_ids import vertex_ids as VERTEX_IDS
        face_vert_ids = np.loadtxt(join(paths.model_data_folder, 'face_lm_vert_inds_SMPL_full_op_ordered.txt'),
                                   delimiter=',',
                                   dtype=int)

        body_model.vertex_joint_selector.extra_joints_idxs = torch.cat([body_model.vertex_joint_selector.extra_joints_idxs, torch.tensor(face_vert_ids)])


    baby_mesh = load_mesh(paths.smil_template_fn, process=False)

    smil_v_template = torch.tensor(baby_mesh.vertices, device=device, dtype=dtype, requires_grad=False)
    smpl_v_template = body_model.v_template.clone().to(device=device).requires_grad_(requires_grad=False)

    body_model.to(device=device)

    orig_posedirs = body_model.posedirs.clone()

    sensor_name = n_utils.sensor_Kinect4Azure
    isinportraitmode = cam_rotation in [90, 270]

    # set up cameras
    cam_list = []
    vis_cam_list = []
    py3d_rasterizer_list = []

    cur_scan_folder = join(data_folder, 'downscaled', 'cropped')
    seq_name = basename(normpath(data_folder))
    k4a_calib_fn = join(cur_scan_folder, "calib.txt")

    depth_cam = n_utils.get_depth_cam_torch(sensor_name, portrait_mode=isinportraitmode,
                                            k4a_filename=k4a_calib_fn)

    focal = torch.tensor([[depth_cam.focal_length_x, depth_cam.focal_length_y]], device=device)
    py3d_cam = PerspectiveCameras(focal_length=focal,
                                  principal_point=depth_cam.center,
                                  device=device,
                                  # R=R, T=T,
                                  image_size=((max(im_wid, im_hgt), max(im_wid, im_hgt)),),
                                  in_ndc=False
                                  )
    # sigma = 1e-4
    raster_settings = RasterizationSettings(
        image_size=max(im_wid, im_hgt),
        blur_radius=0., #np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=1, #40,
        max_faces_per_bin=10000
    )

    py3d_rasterizer = MeshRasterizer(
        cameras=py3d_cam,
        raster_settings=raster_settings
    )

    py3d_rasterizer_list.append(py3d_rasterizer)

    vis_cam_list.append(depth_cam)

    cam_list.append(create_camera(camera_type='persp',
                           focal_length_x=depth_cam.focal_length_x,
                           focal_length_y=depth_cam.focal_length_y,
                           center=depth_cam.center,
                           dtype=dtype))


    conf = n_utils.get_seq_info(data_folder=data_folder,
                                out_folder=output_folder,
                                rotation=cam_rotation)

    if not exists(data_folder):
        print('Folder {} does not exist!!'.format(data_folder))
        return

    save_folder = join(output_folder, '{}_{}_torch'.format(model_type, gender))

    print('Processing {} with {}. Saving results to {}.'.format(data_folder, model_type, save_folder))

    mean_shape_fn = join(save_folder, 'mean_shape.npy')
    mean_model_scale_fn = join(save_folder, 'mean_model_scale.txt')
    mean_model_scale = None

    all_joints_dict = {}
    main_joints_dict = {}
    all_joints_npy = None
    main_joints_npy = None

    mean_shape = None
    init_joints = None

    if not exists(save_folder):
        makedirs(save_folder)

    ### Save all settings
    all_settings = {'weights': opt_weights_dict,
                    'use_hands': use_hands,
                    'use_face': use_face,
                    'use_face_contour': use_face_contour,
                    'use_m2s': use_m2s,
                    'rho_j2d_list': rho_j2d_list,
                    'rho_s2m_list': rho_s2m_list,
                    'rho_m2s_list': rho_m2s_list,
                    'rho_close_plane': rho_close_plane,
                    'pose_prior_type': pose_prior_type,
                    'model_type': model_type,
                    'gender': gender,
                    'poseonly': do_poseonly,
                    'optimize_scale': optimize_scale
                    }
    ###
    filelist = n_utils.get_file_list(cur_scan_folder)

    if do_save_joints:

        body_pose_opt = torch.zeros((1, body_model.NUM_BODY_JOINTS * 3), device=device, dtype=dtype)
        betas_opt = torch.zeros((1, body_model.num_betas), device=device, dtype=dtype,
                                requires_grad=True)
        global_orient_opt = torch.zeros((1, 3), device=device, dtype=dtype,
                                        requires_grad=True)
        transl_opt = torch.zeros((1, 3), device=device, dtype=dtype,
                                 requires_grad=True)

        if model_type == 'smplh' or model_type == 'smplx':
            num_hand_comps = body_model.num_pca_comps if body_model.use_pca else 3 * body_model.NUM_HAND_JOINTS
        else:
            num_hand_comps = 0

        model_output = body_model(betas=torch.zeros((1, body_model.num_betas), device=device, dtype=dtype),
                                  body_pose=torch.zeros((1, body_model.NUM_BODY_JOINTS * 3), device=device, dtype=dtype),
                                  transl= torch.zeros((1, 3), device=device, dtype=dtype),
                                  global_orient=torch.zeros((1, 3), device=device, dtype=dtype),
                                  left_hand_pose=torch.zeros((1, num_hand_comps), device=device, dtype=dtype) if num_hand_comps > 0 else None,
                                  right_hand_pose=torch.zeros((1, num_hand_comps), device=device, dtype=dtype) if num_hand_comps > 0 else None,
                                  return_verts=True,
                                  return_full_pose=True)

        num_joints = model_output.joints.shape[1]
        all_joints_npy = np.ones((len(filelist), num_joints, 3)) * -100.
        main_joints_npy = np.ones((len(filelist), body_model.NUM_BODY_JOINTS + 1, 3)) * -100.
        all_body_poses_npy = np.ones((len(filelist), model_output.body_pose.shape[1])) * -100. # body_model.body_pose
        all_full_poses_npy = np.ones((len(filelist), model_output.full_pose.shape[1])) * -100.
        all_betas_npy = np.ones((len(filelist), model_output.betas.shape[1])) * -100.
        all_centers_of_mass_npy = np.ones((len(filelist), 3)) * -100.

    else:
        # only save config when fitting is performed
        curdatetime = datetime.now()

        timestampStr = curdatetime.strftime("%Y%m%dT%H%M%S.%f")
        conf_fn = join(save_folder, 'conf_{}.yaml'.format(timestampStr))

        with open(conf_fn, 'w') as conf_file:
            yaml.dump(all_settings, conf_file)

    angle_prior = None # create_prior(prior_type='angle', dtype=dtype)

    plane_filename = join(cur_scan_folder, 'bg_plane.gre')
    plane_params = torch.tensor(n_utils.read_plane_parms_from_file(plane_filename), device=device)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        for cur_cam_num in range(len(cam_list)):
            cam_list[cur_cam_num] = cam_list[cur_cam_num].to(device=device)

        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # Cam 0 is always the reference camera/coordinate frame
    camera = cam_list[0]

    init = True

    model_scale_opt = None

    if optimize_scale:
        model_scale_init_val = 0.5
        model_scale_opt = torch.tensor([model_scale_init_val], dtype=dtype, device=device,
                                     requires_grad=True)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        part_segm_fn = paths.smpl_parts_segm_fn

        if exists(part_segm_fn):
            # Read the part segmentation
            part_segm_fn = expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)
        else:
            interpenetration = False
            print('File {} missing! Not using interpenetration term!'.format(part_segm_fn))

    vis_cam = vis_cam_list[0]
    rasterizer = py3d_rasterizer_list[0]

    loss = fitting.RGBDLoss(
        shape_prior=shape_prior,
        body_pose_prior=body_pose_prior,
        use_face=use_face,
        use_hands=use_hands,
        left_hand_prior=left_hand_prior,
        right_hand_prior=right_hand_prior,
        use_joints_conf=True,
        use_s2m=True,
        use_m2s=use_m2s,
        opt_trans=False,
        angle_prior=angle_prior,
        rho_close_plane=rho_close_plane,
        vis_cam=vis_cam,
        rasterizer=rasterizer,
        interpenetration=interpenetration,
        pen_distance=pen_distance,
        search_tree=search_tree,
        tri_filtering_module=filter_faces,
        model_type=model_type
    )

    loss = loss.to(device=device)

    init_loss = fitting.RGBDLoss(
        data_weight=1.,
        rho=rho_j2d_list[0],
        rho_s2m=rho_s2m_list[0],
        rho_m2s=rho_m2s_list[0],
        shape_prior=shape_prior,
        shape_weight= 50.,
        body_pose_prior=body_pose_prior,
        body_pose_weight= 1.,
        use_face=False,
        use_hands=False,
        use_s2m=True,
        s2m_weight= 5e5,
        use_m2s=use_m2s,
        m2s_weight=2e5,
        angle_prior=angle_prior,
        bending_prior_weight=1.,
        rho_close_plane=0.303,
        smooth_reg_weight=0.,
        vis_cam=vis_cam,
        rasterizer=rasterizer,
        interpenetration=interpenetration,
        pen_distance=pen_distance,
        search_tree=search_tree,
        tri_filtering_module=filter_faces,
        model_type=model_type)

    init_loss = init_loss.to(device=device)

    N = body_model.v_template.shape[0]
    m2s_v_weights = torch.ones(N, dtype=dtype, device=device)  # one weight per vertex

    optimizer = None
    opt_create_graph = False

    ### Use initialization frame
    init_frame = 0

    frame_list = [x for x in chain(range(init_frame, len(filelist)), range(0, init_frame+1)[::-1])]

    # create parameters to be optimized
    body_pose_opt = torch.zeros((1, body_model.NUM_BODY_JOINTS * 3), device=device, dtype=dtype,
                                requires_grad=True)
    betas_opt = torch.zeros((1, body_model.num_betas), device=device, dtype=dtype,
                                requires_grad=True)
    global_orient_opt = torch.zeros((1, 3), device=device, dtype=dtype,
                            requires_grad=True)
    transl_opt = torch.zeros((1, 3), device=device, dtype=dtype,
                                    requires_grad=True)

    if model_type == 'smplh' or model_type == 'smplx':
        num_hand_comps = body_model.num_pca_comps if body_model.use_pca else 3 * body_model.NUM_HAND_JOINTS
        left_hand_pose_opt = torch.zeros((1, num_hand_comps),
                                         device=device, dtype=dtype, requires_grad=True)
        right_hand_pose_opt = torch.zeros((1, num_hand_comps),
                                          device=device, dtype=dtype, requires_grad=True)
    else:
        left_hand_pose_opt = None
        right_hand_pose_opt = None
        num_hand_comps = 0

    beta_collection = [] # to create mean shape
    model_scale_collection = []

    if do_poseonly and mean_shape is None:
        if exists(mean_shape_fn):
            mean_shape = np.load(mean_shape_fn)

            betas_opt.requires_grad = False

            betas_opt[:] = torch.tensor(mean_shape)
        if exists(mean_model_scale_fn):
            mean_model_scale = np.loadtxt(mean_model_scale_fn, ndmin=1)
            model_scale_opt = torch.tensor(mean_model_scale, device=device, dtype=dtype,
                                           requires_grad=False)

    joint_diff_to_last_frame = None

    for list_ind, frame_num in enumerate(frame_list):
        if abs(frame_num - frame_list[list_ind-1]) > 1:
            init = True

        if do_poseonly and mean_shape is None and betas_opt.requires_grad and len(beta_collection) == 5:
            # keep shape_fixed and always use mean shape
            if len(model_scale_collection) == 5:
                mean_model_scale = np.array([np.mean(np.array(model_scale_collection).squeeze(), axis=0)])
                model_scale_opt = torch.tensor([mean_model_scale], device=device, dtype=dtype,
                                               requires_grad=False)
                np.savetxt(mean_model_scale_fn, mean_model_scale)


            mean_shape = np.mean(np.array(beta_collection).squeeze(), axis=0)
            np.save(mean_shape_fn, mean_shape)
            betas_opt.requires_grad = False
            betas_opt[:] = torch.tensor(mean_shape)

        filename = filelist[frame_num]
        print('### {} - frame {}/{}: {}'.format(data_folder, frame_num, len(filelist)-1, filename))

        curr_result_fn = join(save_folder, splitext(basename(filename))[0] + '.pkl')

        if exists(curr_result_fn) and exists(join(save_folder, splitext(basename(filename))[0] + '.ply')):
            if do_save_joints:
                with open(curr_result_fn, 'rb') as result_file:
                    loaded_params = pickle.load(result_file)

                if 'model_scale' in loaded_params:
                    cur_scale = loaded_params['model_scale']
                else:
                    cur_scale = 1.

                model_scale_opt = torch.tensor(cur_scale, dtype=dtype, device=device,
                                               requires_grad=True)

                if smil_v_template is not None:
                    body_model.v_template = (smpl_v_template * model_scale_opt) + (
                                (1. - model_scale_opt) * smil_v_template)
                else:
                    body_model.v_template = smpl_v_template * model_scale_opt

                with torch.no_grad():
                    if 'body_pose' in loaded_params:
                        body_pose_opt[:] = torch.tensor(loaded_params['body_pose'])
                    if 'global_orient' in loaded_params:
                        global_orient_opt[:] = torch.tensor(loaded_params['global_orient'])
                    if 'transl' in loaded_params:
                        transl_opt[:] = torch.tensor(loaded_params['transl'])
                    if 'betas' in loaded_params:
                        tmp_betas = loaded_params['betas']
                        betas_opt[:] = torch.tensor(tmp_betas)
                    if 'left_hand_pose' in loaded_params and left_hand_pose_opt is not None and loaded_params['left_hand_pose'] is not None:
                        left_hand_pose_opt[:] = torch.tensor(loaded_params['left_hand_pose'])
                    if 'right_hand_pose' in loaded_params and right_hand_pose_opt is not None and loaded_params['right_hand_pose'] is not None:
                        right_hand_pose_opt[:] = torch.tensor(loaded_params['right_hand_pose'])

                model_output = body_model(betas=betas_opt,
                                          body_pose=body_pose_opt,
                                          transl=transl_opt,
                                          global_orient=global_orient_opt,
                                          left_hand_pose=left_hand_pose_opt,
                                          right_hand_pose=right_hand_pose_opt,
                                          return_verts=True,
                                          return_full_pose=True)

                all_joints_dict[frame_num] = model_output.joints.detach().cpu().numpy().squeeze()
                main_joints_dict[frame_num] = model_output.joints.detach().cpu().numpy().squeeze()[:body_model.NUM_BODY_JOINTS + 1]
                all_joints_npy[frame_num] = model_output.joints.detach().cpu().numpy().squeeze()
                main_joints_npy[frame_num] = model_output.joints.detach().cpu().numpy().squeeze()[:body_model.NUM_BODY_JOINTS + 1]

                all_body_poses_npy[frame_num] = loaded_params['body_pose'].detach().cpu().numpy() if torch.is_tensor(loaded_params['body_pose']) else loaded_params['body_pose']
                all_full_poses_npy[frame_num] = model_output.full_pose.detach().cpu().numpy().squeeze() if torch.is_tensor(model_output.full_pose) else model_output.full_pose
                tmp_betas = loaded_params['betas']
                all_betas_npy[frame_num] = tmp_betas.detach().cpu().numpy().squeeze() if torch.is_tensor(tmp_betas) else tmp_betas

                verts = model_output.vertices.detach().cpu().numpy().squeeze()


                tmp_mesh = Trimesh(vertices=verts,
                                   faces=body_model.faces_tensor.detach().cpu().numpy().squeeze(),
                                   process=False)
                if do_save_plys:
                    mesh_fn = join(save_folder, splitext(basename(filename))[0] + '.ply')
                    tmp_mesh.export(mesh_fn)
                all_centers_of_mass_npy[frame_num] = tmp_mesh.center_mass

            continue

        param_opt_dict = {'body_pose': body_pose_opt,
                          'betas': betas_opt,
                          'transl': transl_opt,
                          'global_orient': global_orient_opt,
                          'left_hand_pose': left_hand_pose_opt,
                          'right_hand_pose': right_hand_pose_opt
                          }

        if init and frame_num != init_frame:
            # load last existing result as initialization
            last_result_fn = join(save_folder, splitext(basename(filelist[frame_list[list_ind-1]]))[0] + '.pkl')

            if exists(last_result_fn):
                with open(last_result_fn, 'rb') as last_result_file:
                    loaded_params = pickle.load(last_result_file)

                if mean_model_scale is not None:
                    model_scale_opt = torch.tensor(mean_model_scale, device=device, dtype=dtype,
                                                   requires_grad=False)
                elif 'model_scale' in loaded_params:
                    model_scale_opt = torch.tensor(loaded_params['model_scale'], dtype=dtype, device=device,
                                                   requires_grad=True)

                if smil_v_template is not None:
                    body_model.v_template = (smpl_v_template * model_scale_opt) + (
                            (1. - model_scale_opt) * smil_v_template)
                else:
                    body_model.v_template = smpl_v_template * model_scale_opt

                with torch.no_grad():
                    if 'body_pose' in loaded_params:
                        body_pose_opt[:] = torch.tensor(loaded_params['body_pose'])
                    if 'global_orient' in loaded_params:
                        global_orient_opt[:] = torch.tensor(loaded_params['global_orient'])
                    if 'transl' in loaded_params:
                        transl_opt[:] = torch.tensor(loaded_params['transl'])
                    if 'betas' in loaded_params:
                        tmp_betas = loaded_params['betas']
                        betas_opt[:] = torch.tensor(tmp_betas)
                    if 'left_hand_pose' in loaded_params and loaded_params['left_hand_pose'] is not None:
                        left_hand_pose_opt[:] = torch.tensor(loaded_params['left_hand_pose'])
                    if 'right_hand_pose' in loaded_params and loaded_params['right_hand_pose'] is not None:
                        right_hand_pose_opt[:] = torch.tensor(loaded_params['right_hand_pose'])

            init = False
        elif init:
            starttime = timer()
            print('*** Starting initial fit... ***')

            body_model, _, _, param_opt_dict = process_one_frame(body_model,
                                                                 param_opt_dict,
                                                                 camera,
                                                                 conf,
                                                                 cur_scan_folder,
                                                                 filename,
                                                                 loss=init_loss,
                                                                 device=device,
                                                                 init=init,
                                                                 body_optimizer=optimizer,
                                                                 body_create_graph=opt_create_graph,
                                                                 use_hands=use_hands,
                                                                 use_face=use_face,
                                                                 model_scale_opt=model_scale_opt,
                                                                 do_viz=do_viz,
                                                                 model_type=model_type,
                                                                 opt_weights_dict=opt_weights_dict,
                                                                 rho_j2d_list=rho_j2d_list,
                                                                 rho_s2m_list=rho_s2m_list,
                                                                 rho_m2s_list=rho_m2s_list,
                                                                 m2s_v_weights=m2s_v_weights,
                                                                 mv=mv,
                                                                 smil_v_template=smil_v_template,
                                                                 smpl_v_template=smpl_v_template,
                                                                 optimize_scale=optimize_scale,
                                                                 plane_params=plane_params,
                                                                 joint_weight_list=joint_weight_list,
                                                                 dtype=dtype,
                                                                 orig_posedirs=orig_posedirs)

            endtime = timer()
            print('Initial fitting took %f s' % (endtime - starttime))

            init = False

            body_model, optimizer, opt_create_graph, param_opt_dict = process_one_frame(body_model,
                                                                                        param_opt_dict,
                                                                                        camera,
                                                                                        conf,
                                                                                        cur_scan_folder,
                                                                                        filename,
                                                                                        loss=loss,
                                                                                        device=device,
                                                                                        init=init,
                                                                                        body_optimizer=optimizer,
                                                                                        body_create_graph=opt_create_graph,
                                                                                        use_hands=use_hands,
                                                                                        use_face=use_face,
                                                                                        model_scale_opt=model_scale_opt,
                                                                                        do_viz=do_viz,
                                                                                        model_type=model_type,
                                                                                        opt_weights_dict=opt_weights_dict,
                                                                                        rho_j2d_list=rho_j2d_list,
                                                                                        rho_s2m_list=rho_s2m_list,
                                                                                        rho_m2s_list=rho_m2s_list,
                                                                                        m2s_v_weights=m2s_v_weights,
                                                                                        mv=mv,
                                                                                        smil_v_template=smil_v_template,
                                                                                        smpl_v_template=smpl_v_template,
                                                                                        optimize_scale=optimize_scale,
                                                                                        plane_params=plane_params,
                                                                                        joint_weight_list=joint_weight_list,
                                                                                        dtype=dtype,
                                                                                        orig_posedirs=orig_posedirs
                                                                                        )

        if do_save_joints:
            print('Frame {} missing. Skipping.'.format(frame_num))
            continue

        joints_pre_opt = body_model(betas=betas_opt,
                                    body_pose=body_pose_opt,
                                    transl=transl_opt,
                                    global_orient=global_orient_opt,
                                    left_hand_pose=left_hand_pose_opt,
                                    right_hand_pose=right_hand_pose_opt,
                                    return_verts=False).joints.detach()

        if init_joints is None:
            init_joints = body_model(betas=betas_opt,
                                     body_pose=body_pose_opt,
                                     transl=transl_opt,
                                     global_orient=global_orient_opt,
                                     left_hand_pose=left_hand_pose_opt,
                                     right_hand_pose=right_hand_pose_opt,
                                     return_verts=False).joints.clone().detach()

        starttime = timer()

        body_model, optimizer, opt_create_graph, param_opt_dict = process_one_frame(body_model,
                                                                                    param_opt_dict,
                                                                                    camera,
                                                                                    conf,
                                                                                    cur_scan_folder,
                                                                                    filename,
                                                                                    loss=loss,
                                                                                    device=device,
                                                                                    init=init,
                                                                                    body_optimizer=optimizer,
                                                                                    body_create_graph=opt_create_graph,
                                                                                    use_hands=use_hands,
                                                                                    use_face=use_face,
                                                                                    model_scale_opt=model_scale_opt,
                                                                                    do_viz=do_viz,
                                                                                    model_type=model_type,
                                                                                    opt_weights_dict=opt_weights_dict,
                                                                                    rho_j2d_list=rho_j2d_list,
                                                                                    rho_s2m_list=rho_s2m_list,
                                                                                    rho_m2s_list=rho_m2s_list,
                                                                                    m2s_v_weights=m2s_v_weights,
                                                                                    joint_diff_to_last_frame=joint_diff_to_last_frame,
                                                                                    mv=mv,
                                                                                    smil_v_template=smil_v_template,
                                                                                    smpl_v_template=smpl_v_template,
                                                                                    optimize_scale=optimize_scale,
                                                                                    plane_params=plane_params,
                                                                                    init_joints=init_joints,
                                                                                    joint_weight_list=joint_weight_list,
                                                                                    dtype=dtype,
                                                                                    orig_posedirs=orig_posedirs
                                                                                    )


        if do_poseonly and mean_shape is None and betas_opt.requires_grad:
            beta_collection.append(betas_opt.detach().cpu().numpy().squeeze())

            if model_scale_opt is not None:
                model_scale_collection.append(model_scale_opt.detach().cpu().numpy().squeeze())

        joint_diff_to_last_frame = body_model(betas=betas_opt,
                                              body_pose=body_pose_opt,
                                              transl=transl_opt,
                                              global_orient=global_orient_opt,
                                              left_hand_pose=left_hand_pose_opt,
                                              right_hand_pose=right_hand_pose_opt,
                                              return_verts=False).joints.detach() - joints_pre_opt

        endtime = timer()
        print('Fitting took %f s' % (endtime - starttime))

        if DO_SAVE:
            result = ({key: val.detach().cpu().numpy()
                       for key, val in body_model.named_parameters()})

            result['body_pose'] = body_pose_opt.detach().cpu().numpy() if body_pose_opt is not None else None
            result['betas'] = betas_opt.detach().cpu().numpy() if betas_opt is not None else None
            result['transl'] = transl_opt.detach().cpu().numpy() if transl_opt is not None else None
            result['global_orient'] = global_orient_opt.detach().cpu().numpy() if global_orient_opt is not None else None
            result['left_hand_pose'] = left_hand_pose_opt.detach().cpu().numpy() if left_hand_pose_opt is not None else None
            result['right_hand_pose'] = right_hand_pose_opt.detach().cpu().numpy() if right_hand_pose_opt is not None else None

            if model_scale_opt is not None:
                result['model_scale'] = model_scale_opt.detach().cpu().numpy().squeeze()

            with open(curr_result_fn, 'wb') as result_file:
                pickle.dump(result, result_file)

            model_output = body_model(return_verts=True,
                                      body_pose=torch.tensor(result['body_pose'], device=device),
                                      transl=torch.tensor(result['transl'], device=device),
                                      global_orient=torch.tensor(result['global_orient'], device=device),
                                      left_hand_pose=torch.tensor(result['left_hand_pose'], device=device) if result['left_hand_pose'] is not None else None,
                                      right_hand_pose=torch.tensor(result['right_hand_pose'], device=device) if result['right_hand_pose'] is not None else None,
                                      betas=torch.tensor(result['betas'], device=device))
            verts = model_output.vertices.detach().cpu().numpy().squeeze()

            alignment = Trimesh(vertices=verts, faces=body_model.faces_tensor.detach().cpu().numpy().squeeze(), process=False)
            mesh_fn = join(save_folder, splitext(basename(filename))[0] + '.ply')
            alignment.export(mesh_fn)

        init = False

    if do_save_joints:
        if -100. in all_joints_npy:
            print('*********** MISSING VALUES!??!? Finish processing of sequence before running with --saveonly! **************')
            sys.exit()

        with open(join(save_folder, '{}_predicted_joints_all.txt'.format(seq_name)), 'w') as outfile:
            for fr_ind in range(len(filelist)):
                if fr_ind in all_joints_dict:
                    val = all_joints_dict[fr_ind]
                    outfile.write('%i %i\n' % (fr_ind, val.shape[0]))

                    for joint_i, pos_ji in enumerate(val):
                        outfile.write('%f %f %f %i\n' % (pos_ji[0], pos_ji[1], pos_ji[2], joint_i))

        with open(join(save_folder, '{}_predicted_joints_main.txt'.format(seq_name)), 'w') as outfile:
            for fr_ind in range(len(filelist)):
                if fr_ind in main_joints_dict:
                    val = main_joints_dict[fr_ind]
                    outfile.write('%i %i\n' % (fr_ind, val.shape[0]))

                    for joint_i, pos_ji in enumerate(val):
                        outfile.write('%f %f %f %i\n' % (pos_ji[0], pos_ji[1], pos_ji[2], joint_i))

        np.save(join(save_folder, '{}_predicted_joints_all.npy'.format(seq_name)), all_joints_npy.astype(np.float32))
        np.save(join(save_folder, '{}_predicted_joints_main.npy'.format(seq_name)), main_joints_npy.astype(np.float32))

        np.save(join(save_folder, '{}_all_body_poses.npy'.format(seq_name)), all_body_poses_npy.astype(np.float32))
        np.savetxt(join(save_folder, '{}_all_body_poses.txt'.format(seq_name)), all_body_poses_npy.astype(np.float32))
        np.save(join(save_folder, '{}_all_full_poses.npy'.format(seq_name)), all_full_poses_npy.astype(np.float32))
        np.savetxt(join(save_folder, '{}_all_full_poses.txt'.format(seq_name)), all_full_poses_npy.astype(np.float32))
        np.savetxt(join(save_folder, '{}_all_betas.txt'.format(seq_name)), all_betas_npy.astype(np.float32))
        np.savetxt(join(save_folder, '{}_all_centers_of_mass.txt'.format(seq_name)), all_centers_of_mass_npy.astype(np.float32))

    if mv is not None:
        mv.close_viewer()
