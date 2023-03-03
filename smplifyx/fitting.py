# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#
# Modifications were made by Nikolas Hesse, University Children’s Hospital Zurich – Eleonore Foundation

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
from pytorch3d.structures.meshes import Meshes

from psbody.mesh import Mesh

from mesh_viewer import MeshViewer
import utils

from custom_utils.n_utils import get_foot_sole_v_ids, get_signed_dists_to_plane, \
    py3d_point_mesh_face_distance_w_inds, get_vertex_weights_wrt_face_area


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl', viewer_flags={},
                 mv=None, **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        self.viewer_flags = viewer_flags
        self.mv = mv

    def __enter__(self):
        self.steps = 0
        if self.visualize and self.mv is None:
            self.mv = MeshViewer(body_color=self.body_color, viewer_flags=self.viewer_flags)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize and False:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model, scan=None,
                    param_opt_dict=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl'
        prev_loss = None
        for n in range(self.maxiters):

            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

                if param_opt_dict is not None:
                    model_output = body_model(global_orient=param_opt_dict['global_orient'],
                                              body_pose=param_opt_dict['body_pose'],
                                              betas=param_opt_dict['betas'],
                                              transl=param_opt_dict['transl'],
                                              return_verts=True)
                else:
                    model_output = body_model(
                        return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces_tensor.detach().cpu().numpy().squeeze())

                if scan is not None:
                    self.mv.update_pointcloud(scan.vertices)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer,
                               body_model,
                               camera=None,
                               gt_joints=None,
                               loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_full_pose=False,
                               model_scale_opt=None,
                               create_graph=False,
                               init=False,
                               scan_v_tensor=None,
                               depth_im=None,
                               prev_trans=None,
                               prev_pose_dict=None,
                               smil_v_template=None,
                               smpl_v_template=None,
                               plane_params=None,
                               param_opt_dict=None,
                               orig_posedirs=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl'

        body_pose_opt = param_opt_dict['body_pose'] if 'body_pose' in param_opt_dict else None
        global_orient_opt = param_opt_dict['global_orient'] if 'global_orient' in param_opt_dict else None
        transl_opt = param_opt_dict['transl'] if 'transl' in param_opt_dict else None
        betas_opt = param_opt_dict['betas'] if 'betas' in param_opt_dict else None
        left_hand_pose_opt = param_opt_dict['left_hand_pose'] if 'left_hand_pose' in param_opt_dict else None
        right_hand_pose_opt = param_opt_dict['right_hand_pose'] if 'right_hand_pose' in param_opt_dict else None

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            if model_scale_opt is not None and smpl_v_template is not None:
                # interpolate between SMIL and SMPL
                if smil_v_template is not None:
                    body_model.v_template = (smpl_v_template * model_scale_opt) + (
                            (1. - model_scale_opt) * smil_v_template)
                else:
                    body_model.v_template = smpl_v_template * model_scale_opt

                with torch.no_grad():
                    body_model.posedirs = orig_posedirs * (((1 - model_scale_opt) * 0.35) + (model_scale_opt * 1.)) # scale posedirs to model scale

            body_model_output = body_model(betas=betas_opt,
                                           body_pose=body_pose_opt,
                                           transl=transl_opt,
                                           global_orient=global_orient_opt,
                                           left_hand_pose=left_hand_pose_opt,
                                           right_hand_pose=right_hand_pose_opt,
                                           return_verts=True,
                                           return_full_pose=return_full_pose)


            mesh_depth_visibility = None

            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              model_scale_opt=model_scale_opt,
                              visualize=self.visualize,
                              mesh_depth_visibility=mesh_depth_visibility,
                              init=init,
                              scan_v_tensor=scan_v_tensor,
                              depth_im=depth_im,
                              prev_trans=prev_trans,
                              prev_pose_dict=prev_pose_dict,
                              plane_params=plane_params,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(betas=betas_opt,
                                           body_pose=body_pose_opt,
                                           transl=transl_opt,
                                           global_orient=global_orient_opt,
                                           left_hand_pose=left_hand_pose_opt,
                                           right_hand_pose=right_hand_pose_opt,
                                           return_verts=True)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces_tensor.detach().cpu().numpy().squeeze())

                if scan_v_tensor is not None:
                    self.mv.update_pointcloud(scan_v_tensor.detach().cpu().numpy()[0])

            return total_loss

        return fitting_func


class RGBDLoss(nn.Module):

    def __init__(self,
                 dtype=torch.float32,
                 data_weight=1.,
                 use_joints_conf=True,
                 rho_j2d=100,
                 rho_s2m=1.,
                 rho_m2s=1.,
                 shape_prior=None,
                 shape_weight=1.,
                 body_pose_prior=None,
                 body_pose_weight=1.,
                 use_hands=False,
                 use_face=False,
                 left_hand_prior=None,
                 right_hand_prior=None,
                 hand_prior_weight=1.,
                 use_s2m=False,
                 s2m_weight=0.0,
                 use_m2s=False,
                 m2s_weight=0.0,
                 opt_trans=False,
                 trans_reg_weight=0.0,
                 smooth_reg_weight=0.,
                 angle_prior=None,
                 bending_prior_weight=0.,
                 inside_plane_weight=0.,
                 close_plane_weight=0.,
                 rho_close_plane=1e-2,
                 vis_cam=None,
                 rasterizer=None,
                 interpenetration=True,
                 search_tree=None,
                 pen_distance=None,
                 tri_filtering_module=None,
                 coll_loss_weight=0.0,
                 model_type='smplh',
                 **kwargs):

        super(RGBDLoss, self).__init__()

        self.cnt = 0
        self.vis = None
        self.vis_f = None

        self.vis_cam = vis_cam
        self.rasterizer = rasterizer

        self.use_joints_conf = use_joints_conf

        self.j2d_gmof = utils.GMoF(rho=rho_j2d)
        self.s2m_gmof = utils.GMoF(rho=rho_s2m)
        self.m2s_gmof = utils.GMoF(rho=rho_m2s)

        self.shape_prior = shape_prior
        self.body_pose_prior = body_pose_prior

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face

        self.model_type = model_type

        self.use_s2m = use_s2m
        self.use_m2s = use_m2s
        self.opt_trans = opt_trans
        self.angle_prior = angle_prior

        self.outside_mask = None

        self.register_buffer('inside_plane_weight',
                             torch.tensor(inside_plane_weight, dtype=dtype))
        self.close_plane_gmof = utils.GMoF(rho=rho_close_plane)
        self.inside_plane_gmof = utils.GMoF(rho=.2)

        self.register_buffer('close_plane_weight',
                             torch.tensor(close_plane_weight, dtype=dtype))

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        self.register_buffer('s2m_weight',
                             torch.tensor(s2m_weight, dtype=dtype))
        self.register_buffer('m2s_weight',
                             torch.tensor(m2s_weight, dtype=dtype))
        if self.opt_trans:
            self.register_buffer('trans_reg_weight',
                                 torch.tensor(trans_reg_weight, dtype=dtype))

        self.register_buffer('smooth_reg_weight',
                             torch.tensor(smooth_reg_weight, dtype=dtype))

        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))

        self.all_pose = []

        self.interpenetration = interpenetration

        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        self.foot_sole_v_ids = get_foot_sole_v_ids(model_type=model_type)

        self.needs_transl_reset = False

        self.R_to_py3d = torch.tensor(
            [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=rasterizer.cameras.device
        )

        self.vertex_weights_wrt_face_area = get_vertex_weights_wrt_face_area(model_type=model_type,
                                                                             device=rasterizer.cameras.device,
                                                                             dtype=dtype)

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

        self.cnt = 0
        self.vis = None

    def reset_gmo_rhos(self, rho_j2d, rho_s2m, rho_m2s):
        self.j2d_gmof = utils.GMoF(rho=rho_j2d)
        self.s2m_gmof = utils.GMoF(rho=rho_s2m)
        self.m2s_gmof = utils.GMoF(rho=rho_m2s)

    # @profile
    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights, body_model=None,
                prev_trans=None, prev_pose_dict=None,
                visualize=False,
                s2m_obj=None, m2s_obj=None, m2s_v_weights=None,
                scan_v_tensor=None, scan_py3d_cloud=None, mesh_depth_visibility=None,
                init=False,
                joint_diff_to_last_frame=None, plane_params=None,
                **kwargs):

        loss_list = {}

        projected_joints = camera(body_model_output.joints).to(device=gt_joints.device)

        # Calculate the loss from the Pose prior
        use_gmm_prior = True

        if use_gmm_prior:
            pprior_loss = torch.sum(self.body_pose_prior(body_model_output.body_pose,
                                                         body_model_output.betas)) * self.body_pose_weight ** 2

        loss_list['pprior_loss'] = pprior_loss.item()
        total_loss = pprior_loss

        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections

        if torch.sum(weights) > 0.:
            joint_diff = gt_joints - projected_joints
            joint_diff = self.j2d_gmof(joint_diff)
            joint_loss = (118. * torch.mean(weights ** 2 * joint_diff, [1, 2])) * self.data_weight ** 2
            loss_list['joint_loss'] = joint_loss.item()

            total_loss += joint_loss.squeeze()

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None and (self.model_type == 'smplx' or self.model_type == 'smplh'):
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(body_model_output.left_hand_pose)) * self.hand_prior_weight ** 2

            loss_list['left_hand_prior_loss'] = left_hand_prior_loss.item()
            total_loss += left_hand_prior_loss

        if self.use_hands and self.right_hand_prior is not None and (self.model_type == 'smplx' or self.model_type == 'smplh'):
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(body_model_output.right_hand_pose)) * self.hand_prior_weight ** 2

            loss_list['right_hand_prior_loss'] = right_hand_prior_loss.item()
            total_loss += right_hand_prior_loss


        if self.opt_trans:
            trans_reg = self.trans_reg_weight ** 2 * torch.mean((body_model.trans -  # curr_trans
                                                                 prev_trans).pow(2))  # [:, 2]
            loss_list['trans_reg'] = trans_reg.item()

        if body_model_output.betas.requires_grad:
            shape_loss = torch.sum(self.shape_prior(body_model_output.betas)) * self.shape_weight
            loss_list['shape_loss'] = shape_loss.item()
            total_loss += shape_loss

        if not init and prev_pose_dict is not None and self.smooth_reg_weight > 0.:
            if True and joint_diff_to_last_frame is not None:
                smooth_reg_loss = self.smooth_reg_weight ** 2 * \
                                  torch.sum((torch.norm(body_model_output.joints - \
                                                        (prev_pose_dict['joints'] + \
                                                         (joint_diff_to_last_frame / 2.)), dim=-1)
                                            ) ** 2)

                if 'left_hand_pose' in prev_pose_dict and 'right_hand_pose' in prev_pose_dict:
                    smooth_reg_loss += self.smooth_reg_weight ** 2 * \
                                      torch.sum((torch.abs(body_model_output.left_hand_pose - \
                                                            prev_pose_dict['left_hand_pose']) + \
                                                            torch.abs(body_model_output.right_hand_pose - \
                                                            prev_pose_dict['right_hand_pose'])
                                                 ) ** 2)
            else:
                smooth_reg_loss = self.smooth_reg_weight ** 2 * \
                                  torch.sum((torch.norm(body_model_output.joints - \
                                                                         prev_pose_dict['joints'], dim=-1)) ** 2)

            loss_list['smooth_reg_loss'] = smooth_reg_loss.item()
            total_loss += smooth_reg_loss

        if self.angle_prior:
            body_pose = body_model_output.full_pose[:, 3:66]
            angle_prior_loss = torch.sum(self.angle_prior(body_pose)) * self.bending_prior_weight ** 2
            loss_list['angle_prior_loss'] = angle_prior_loss.item()
            total_loss += angle_prior_loss

        vertices = body_model_output.vertices

        s2m_dist = None
        m2s_dist = None

        if (self.use_s2m or self.use_m2s) and (
                self.s2m_weight > 0. or self.m2s_weight > 0.) and scan_v_tensor is not None:

            if self.vis is None or (self.cnt % 10) == 0 or init or not (True in self.vis):
                if self.rasterizer is not None:

                    py3d_mesh = Meshes(verts=vertices,
                                       faces=body_model_faces.reshape(1,-1,3))
                    fragments = self.rasterizer(py3d_mesh, R=self.R_to_py3d)
                    pix_to_face = fragments.pix_to_face

                    packed_faces = py3d_mesh.faces_packed()
                    packed_verts = py3d_mesh.verts_packed()
                    vertex_visibility_map = torch.zeros(packed_verts.shape[0])  # (V,)
                    visible_faces = pix_to_face.unique()[1:]  # first item = -1 #[0]  # (num_visible_faces )
                    visible_verts_idx = packed_faces[visible_faces]  # (num_visible_faces,  3)
                    unique_visible_verts_idx = torch.unique(visible_verts_idx)  # (num_visible_verts, )

                    vertex_visibility_map[unique_visible_verts_idx] = 1.0

                    old_to_new_indices = np.zeros(vertices.shape[1])
                    old_to_new_indices[unique_visible_verts_idx.detach().cpu().numpy()] = range(unique_visible_verts_idx.size()[0])
                    vis_f = old_to_new_indices[body_model_faces.detach().cpu().numpy().reshape(-1,3)[visible_faces.detach().cpu().numpy()]].astype(np.int32)
                    self.vis_f = vis_f
                    self.vis = vertex_visibility_map.bool()

                else:
                    vertices_np = vertices.detach().cpu().numpy().squeeze()
                    body_faces_np = body_model_faces.detach().cpu().numpy().reshape(-1, 3)

                    m = Mesh(v=vertices_np, f=body_faces_np)

                    nv = vertices.shape[1]

                    self.vis_cam.origin = np.array([0., 0., 0.])

                    (temp_vis, normals) = m.vertex_visibility_and_normals(self.vis_cam, omni_directional_camera=True)

                    temp_vis = temp_vis.squeeze()
                    temp_vis = temp_vis[:nv]

                    self.vis = temp_vis

            self.cnt += 1

            m2s_dist = None
            m2s_inds = None
            ignore_gmof = False

            ################### s2m
            if s2m_obj is None:
                if self.s2m_weight > 0.:

                    if True in self.vis and scan_py3d_cloud is not None:
                        py3d_mesh_vis = Meshes(verts=vertices[:, self.vis],
                                               faces=torch.tensor(self.vis_f, device=vertices.device).reshape(1, -1, 3))

                        s2m_dist, m2s_dist, s2m_inds, m2s_inds = py3d_point_mesh_face_distance_w_inds(py3d_mesh_vis,
                                                                                                      scan_py3d_cloud)

                        # Calculate the loss due to interpenetration
                        if (self.interpenetration and self.coll_loss_weight.item() > 0):
                            batch_size = 1  # projected_joints.shape[0]
                            triangles = torch.index_select(
                                body_model_output.vertices, 1,
                                body_model_faces).view(batch_size, -1, 3, 3)

                            with torch.no_grad():
                                collision_idxs = self.search_tree(triangles)

                            # Remove unwanted collisions
                            if self.tri_filtering_module is not None:
                                collision_idxs = self.tri_filtering_module(collision_idxs)

                            if collision_idxs.ge(0).sum().item() > 0:
                                pen_loss = torch.sum(
                                    self.coll_loss_weight *
                                    self.pen_distance(triangles, collision_idxs)).sqrt()

                                loss_list['pen_loss'] = pen_loss.item()

                                if torch.isnan(pen_loss):
                                    print('asdf')

                                total_loss += pen_loss

                    else:
                        print('Mesh empty or cloud None ????')

                        trans_reg = 100. * torch.mean((torch.mean(body_model_output.vertices, dim=1) - prev_trans).pow(2))  # [:, 2]
                        loss_list['trans_reg'] = trans_reg.item()
                        total_loss += trans_reg


                        py3d_mesh = Meshes(verts=vertices,
                                           faces=body_model_faces.reshape(1, -1, 3))

                        s2m_dist, m2s_dist, s2m_inds, m2s_inds = py3d_point_mesh_face_distance_w_inds(py3d_mesh,
                                                                                                      scan_py3d_cloud)

                        ignore_gmof = False

                    if s2m_dist is not None:
                        if not ignore_gmof:
                            s2m_dist = self.s2m_gmof(s2m_dist.sqrt())
                        s2m_dist = self.s2m_weight * s2m_dist.sum() / s2m_dist.shape[1]

            else:
                if self.s2m_weight > 0.:
                    s2m_dist = self.s2m_weight * (s2m_obj.forward(vertices)).pow(2).sum()

            if s2m_dist is not None:
                loss_list['s2m_dist'] = s2m_dist.item()
                total_loss += s2m_dist

            if self.use_m2s and m2s_dist is not None:
                if m2s_obj is None:
                    if self.m2s_weight > 0 and self.vis.sum() > 0:

                        # Only for m2s!
                        if mesh_depth_visibility is not None:
                            # for m2s
                            if m2s_dist is not None:
                                m2s_dist = m2s_dist[:, mesh_depth_visibility[self.vis.astype(bool)].astype(bool)]

                            self.vis *= mesh_depth_visibility

                        if m2s_dist is not None:
                            if not ignore_gmof:
                                m2s_dist = self.m2s_gmof(m2s_dist.sqrt())

                            if self.vertex_weights_wrt_face_area is not None and self.vis_f.ndim > 1 and m2s_dist.dim() > 1:
                                # m2s contains distance between FACES and points
                                # take mean over vertices per face
                                try:
                                    m2s_dist = m2s_dist * torch.mean(self.vertex_weights_wrt_face_area[self.vis_f], dim=1)
                                except IndexError:
                                    print('IndexError m2s_dist: {}, self.vis_f: {}, self.vertex_weights_wrt_face_area: {}'.format(
                                        m2s_dist, self.vis_f, self.vertex_weights_wrt_face_area))

                            m2s_dist = self.m2s_weight * m2s_dist.sum() / m2s_dist.shape[1]

                        if m2s_dist is not None:
                            loss_list['m2s_dist'] = m2s_dist.item()
                            total_loss += m2s_dist

                else:
                    if self.m2s_weight > 0:
                        m2s_dist = m2s_obj.forward(vertices).pow(2)

                        if self.vertex_weights_wrt_face_area is not None and True in self.vis:
                            m2s_dist = m2s_dist * self.vertex_weights_wrt_face_area[self.vis]

                        m2s_dist = self.m2s_weight * m2s_dist.sum()

                        loss_list['m2s_dist'] = m2s_dist.item()
                        total_loss += m2s_dist


        if self.inside_plane_weight > 0.:
            foot_only = True
            if foot_only:
                inside_plane_loss = self.inside_plane_weight * torch.sum(
                    (torch.clamp(get_signed_dists_to_plane(vertices[0, self.foot_sole_v_ids],
                                                                 plane_params=plane_params,
                                                                 buffer=0.005,
                                                                 do_flip=False),
                                 min=-1., max=0.) ** 2)) / self.foot_sole_v_ids.size
            else:
                inside_plane_loss = self.inside_plane_weight * torch.sum(
                    (torch.clamp(get_signed_dists_to_plane(vertices[0],
                                                                 plane_params=plane_params,
                                                                 buffer=0.005,
                                                                 do_flip=False),
                                 min=-1., max=0.) ** 2)) / vertices.shape[1]
            loss_list['inside_plane_loss'] = inside_plane_loss.item()
            total_loss += inside_plane_loss

        if self.close_plane_weight > 0.:
            close_plane_loss = self.close_plane_weight * torch.sum(
                self.close_plane_gmof(torch.abs(get_signed_dists_to_plane(vertices[0, self.foot_sole_v_ids],  # self.close_plane_gmof
                                                      plane_params=plane_params,
                                                      buffer=0.005,
                                                      do_flip=False)))).pow(2)

            loss_list['close_plane_loss'] = close_plane_loss.item()
            total_loss += close_plane_loss

        if visualize:
            print('total:{:.4f} - {}'.format(total_loss.item(),
                                             ' '.join(['{}:{:0.4f}'.format(k, v) for k, v in loss_list.items()])))

        return total_loss
