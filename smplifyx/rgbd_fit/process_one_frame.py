# Copyright©2023 University Children’s Hospital Zurich – Eleonore Foundation
# By using this code you agree to the terms specified in the LICENSE file

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import torch
from os.path import basename, join, exists

from pytorch3d.structures.pointclouds import Pointclouds
from timeit import default_timer as timer

import fitting
from custom_utils import n_utils
from data_parser import read_keypoints
from optimizers import optim_factory

MAX_ITERS = 1000
LEARNING_RATE = 1.
TOLERANCE = 5e-3
OPTIMIZER = 'lbfgsls'

def process_one_frame(bm=None,
                      param_opt_dict=None,
                      depth_cam=None,
                      conf=None,
                      scan_folder='',
                      filename='',
                      loss=None,
                      device=torch.device('cpu'),
                      init=False,
                      body_optimizer=None,
                      body_create_graph=False,
                      use_hands=False,
                      use_face=False,
                      do_viz=False,
                      model_type='smpl',
                      opt_weights_dict=None,
                      rho_j2d_list=None,
                      rho_s2m_list=None,
                      rho_m2s_list=None,
                      m2s_v_weights=None,
                      joint_diff_to_last_frame=None,
                      mv=None,
                      model_scale_opt=None,
                      optimize_scale=False,
                      smil_v_template=None,
                      smpl_v_template=None,
                      plane_params=None,
                      init_joints=None,
                      joint_weight_list=None,
                      dtype=torch.float32,
                      smooth_addition='',
                      orig_posedirs=None
                      ):

    body_pose_opt = param_opt_dict['body_pose'] if 'body_pose' in param_opt_dict else None
    global_orient_opt = param_opt_dict['global_orient'] if 'global_orient' in param_opt_dict else None
    transl_opt = param_opt_dict['transl'] if 'transl' in param_opt_dict else None
    betas_opt = param_opt_dict['betas'] if 'betas' in param_opt_dict else None
    left_hand_pose_opt = param_opt_dict['left_hand_pose'] if 'left_hand_pose' in param_opt_dict else None
    right_hand_pose_opt = param_opt_dict['right_hand_pose'] if 'right_hand_pose' in param_opt_dict else None

    cloud_filename = basename(filename)

    depth_im, rgb_im, pc_v, pc_v_inds = n_utils.load_frame_torch(scan_folder, cloud_filename, conf['scan_infos'],
                                                                 dtype=dtype, load_rgb=False)

    op_filename = join(scan_folder,
                   'openpose/' + cloud_filename[
                                 :cloud_filename.rfind('_depth')] + '_keypoints' + smooth_addition + '.json')

    if exists(op_filename):
        keypoint_data = read_keypoints(op_filename, use_hands=use_hands, use_face=use_face)
        keypoint_data = torch.tensor(keypoint_data.keypoints[0], dtype=dtype)
    else:
        print('No keypoints!!')
        keypoint_data = torch.zeros((bm.joint_mapper.joint_maps.size()[0], 3), dtype=dtype)


    if pc_v.shape[0] < 1000.: #scan.vertices.size < 1000:
        print('Only {} points in scan!! Skipping.'.format(pc_v.shape[0])) #scan.vertices.size))

        return bm, body_optimizer, body_create_graph


    scan_v_tensor = torch.tensor(pc_v, dtype=dtype, device=device).unsqueeze(0).contiguous()


    scan_py3d_cloud = Pointclouds(points=scan_v_tensor)

    if init:
        global_orient = torch.zeros([1, 3], dtype=torch.float32)

        # shoulders
        if (keypoint_data[2, 0] > keypoint_data[5, 0]):
            print('Back view init!')
            conf['scan_infos']['use_back_view'] = True
            global_orient[0, 2] = 3.1415
        else:
            global_orient[0, 0] = 3.1415

        init_trans = torch.mean(scan_v_tensor, 1)
        init_trans[0, 1] -= 0.15

        body_pose = loss.body_pose_prior.get_mean()

        if body_pose is None:
            body_pose = torch.zeros([1, bm.NUM_BODY_JOINTS * 3], dtype=torch.float32)

        with torch.no_grad():
            body_pose_opt[:] = torch.tensor(body_pose)
            global_orient_opt[:] = torch.tensor(global_orient)
            transl_opt[:] = torch.tensor(init_trans)
        iters = 1000
    else:
        iters = MAX_ITERS

    gt_joints = keypoint_data[:, :2]
    joints_conf = keypoint_data[:, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    joints_conf = joints_conf.to(device=device, dtype=dtype)

    viewer_flags = {'view_center': np.mean(pc_v, axis=0) * [1., 1., -1]}

    if mv is not None:
        mv.viewer.render_lock.acquire()
        for k, v in viewer_flags.items():
            mv.viewer.viewer_flags[k] = v
        mv.viewer.render_lock.release()

    with fitting.FittingMonitor(batch_size=1,
                                visualize=do_viz,
                                gt_joints=gt_joints,
                                maxiters=iters,
                                ftol=TOLERANCE,
                                model_type=model_type,
                                viewer_flags=viewer_flags,
                                mv=mv) as monitor:

        keys = opt_weights_dict.keys()
        opt_weights = [dict(zip(keys, vals)) for vals in
                       zip(*(opt_weights_dict[k] for k in keys
                             if opt_weights_dict[k] is not None))]

        for weight_list in opt_weights:
            for key in weight_list:
                weight_list[key] = torch.tensor(weight_list[key],
                                                device=device,
                                                dtype=dtype)

        orig_betas_reqgrad = betas_opt.requires_grad
        orig_lhand_pose_reqgrad = left_hand_pose_opt.requires_grad
        orig_rhand_pose_reqgrad = right_hand_pose_opt.requires_grad

        for opt_idx, curr_weights in enumerate(opt_weights):

            if opt_idx == 0 and init:
                # first stage trans and global orient only
                body_pose_opt.requires_grad = False
                betas_opt.requires_grad = False
                left_hand_pose_opt.requires_grad = False
                right_hand_pose_opt.requires_grad = False
            else:
                body_pose_opt.requires_grad = True
                betas_opt.requires_grad = orig_betas_reqgrad
                left_hand_pose_opt.requires_grad = orig_lhand_pose_reqgrad
                right_hand_pose_opt.requires_grad = orig_rhand_pose_reqgrad

            if opt_idx < 2: # old 1
                if not init:
                    continue

            print('Fitting stage {}'.format(opt_idx))

            starttime_stage = timer()
            # body_params = [param for param_name, param in bm.named_parameters()]

            body_params = [body_pose_opt, transl_opt, global_orient_opt]

            opt_param_names = []
            if body_pose_opt.requires_grad:
                opt_param_names.append('body_pose')
            if transl_opt.requires_grad:
                opt_param_names.append('translation')
            if global_orient_opt.requires_grad:
                opt_param_names.append('global orientation')

            if betas_opt is not None:
                body_params.append(betas_opt)
                if betas_opt.requires_grad:
                    opt_param_names.append('betas')
            if left_hand_pose_opt is not None and right_hand_pose_opt is not None:
                body_params.append(left_hand_pose_opt)
                body_params.append(right_hand_pose_opt)
                if left_hand_pose_opt.requires_grad and right_hand_pose_opt.requires_grad:
                    opt_param_names.append('hand poses')

            final_params = list(
                filter(lambda x: x.requires_grad, body_params))

            if optimize_scale and model_scale_opt is not None:
                final_params.append(model_scale_opt)
                if model_scale_opt.requires_grad:
                    opt_param_names.append('model interpolation')


            print('Optimizing parameters *** {} ***'.format(', '.join(opt_param_names)))
            if True or body_optimizer is None:
                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params,
                                                                                   optim_type=OPTIMIZER,
                                                                                   maxiters=iters,
                                                                                   lr=LEARNING_RATE,
                                                                                   #ftol=1e-5,
                                                                                   ftol=curr_weights['tolerance'])
                body_optimizer.zero_grad()

            #print('before: {}'.format(bm.trans.detach().cpu().numpy()))

            prev_trans = transl_opt.clone().detach()

            bm_output = bm(betas=betas_opt,
                           body_pose=body_pose_opt,
                           transl=transl_opt,
                           global_orient=global_orient_opt,
                           left_hand_pose=left_hand_pose_opt,
                           right_hand_pose=right_hand_pose_opt,
                           return_verts=True,
                           return_full_pose=True)

            prev_pose_dict = {'betas': betas_opt.clone().detach(),
                              'transl': transl_opt.clone().detach(),
                              'body_pose': bm_output.body_pose.clone().detach(),
                              'global_orient': bm_output.global_orient.clone().detach(),
                              'joints': bm_output.joints.clone().detach()
                              }

            if use_hands and (model_type == 'smplx' or model_type == 'smplh'):
                prev_pose_dict['left_hand_pose'] = bm_output.left_hand_pose.clone().detach()
                prev_pose_dict['right_hand_pose'] = bm_output.right_hand_pose.clone().detach()

            if init_joints is not None:
                prev_pose_dict['init_joints'] = init_joints

            s2m_obj = None

            joint_weights = torch.ones_like(joints_conf)

            # increase eye and ear weights (especially with people wearing masks...)
            # vertex ids added also for SMPL and SMPL-H
            joint_weights[:,15:19] = 3.

            # only use body pose for initialization(?)
            if init:
                joint_weights[:, 25:] = 0.
            else:
                if use_hands:
                    if 'hand_joints_weights' in curr_weights.keys():
                        joint_weights[:, 25:67] = curr_weights['hand_joints_weights']
                    else:
                        joint_weights[:, 25:67] = 2.
                if use_face:
                    joint_weights[:, 67:] = 2.

            if joint_weight_list is not None:
                for joint_id, joint_weight in joint_weight_list:
                    joint_weights[:, int(joint_id)] = joint_weight

            if init:
                tmp_dict = {'smooth_reg_weight': curr_weights['smooth_reg_weight'],
                            'close_plane_weight': curr_weights['close_plane_weight'],
                            'coll_loss_weight': curr_weights['coll_loss_weight'],
                            'shape_weight': curr_weights['shape_weight'],
                            'm2s_weight': curr_weights['m2s_weight']
                            }

                loss.reset_loss_weights(tmp_dict)
            else:
                loss.reset_loss_weights(curr_weights)
                loss.reset_gmo_rhos(rho_j2d=rho_j2d_list[opt_idx],
                                    rho_s2m=rho_s2m_list[opt_idx],
                                    rho_m2s=rho_m2s_list[opt_idx])

            if curr_weights['foot_m2s_weight'] != 1.:
                lfoot_v_ids, rfoot_v_ids = n_utils.get_foot_v_ids(bm)
                m2s_v_weights[lfoot_v_ids] = curr_weights['foot_m2s_weight']
                m2s_v_weights[rfoot_v_ids] = curr_weights['foot_m2s_weight']

            closure = monitor.create_fitting_closure(
                body_optimizer, bm,
                camera=depth_cam, gt_joints=gt_joints,
                joints_conf=joints_conf,
                joint_weights=joint_weights,
                scan_v_tensor=scan_v_tensor,
                scan_py3d_cloud=scan_py3d_cloud,
                loss=loss, create_graph=body_create_graph,
                return_verts=True, return_full_pose=True, viz_mode='mv',
                R_tensor=None, t_tensor=None,
                prev_trans=prev_trans,
                prev_pose_dict=prev_pose_dict,
                s2m_obj=s2m_obj, m2s_obj=None, m2s_v_weights=m2s_v_weights,
                model_scale_opt=model_scale_opt,
                smil_v_template=smil_v_template,
                smpl_v_template=smpl_v_template,
                misspix_mask=(depth_im==0),
                depth_im=depth_im,
                init=init,
                joint_diff_to_last_frame=joint_diff_to_last_frame,
                plane_params=plane_params,
                param_opt_dict=param_opt_dict,
                orig_posedirs=orig_posedirs)

            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                bm,
                param_opt_dict=param_opt_dict)

            endtime_stage = timer()

            print('Stage {} done in {:.2f} seconds. Final loss val: {:.2f}'.format(opt_idx, endtime_stage - starttime_stage,
                                                                           final_loss_val))

    return bm, body_optimizer, body_create_graph, param_opt_dict
