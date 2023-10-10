# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
import queue
import threading

from internal import math, utils  # pylint: disable=g-multiple-import
import jax
import numpy as np
from PIL import Image

import imageio

import cv2
from pathlib import Path


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'zed2': LLFF,
  }
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_to_ndc(origins, directions, focal, width, height, near=1.,
                   focaly=None):
  """Convert a set of rays to normalized device coordinates (NDC).

  Args:
    origins: np.ndarray(float32), [..., 3], world space ray origins.
    directions: np.ndarray(float32), [..., 3], world space ray directions.
    focal: float, focal length.
    width: int, image width in pixels.
    height: int, image height in pixels.
    near: float, near plane along the negative z axis.
    focaly: float, Focal for y axis (if None, equal to focal).

  Returns:
    origins_ndc: np.ndarray(float32), [..., 3].
    directions_ndc: np.ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0; this allows us to share
  a common NDC space for "forward facing" scenes.

  Note that
      projection(origins + t * directions)
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """
  # Shift ray origins to near plane, such that oz = -near.
  # This makes the new near bound equal to 0.!!!! or it would be [-1, 1]
  t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions

  dx, dy, dz = np.moveaxis(directions, -1, 0)
  ox, oy, oz = np.moveaxis(origins, -1, 0)

  fx = focal
  fy = focaly if (focaly is not None) else focal

  # Perspective projection into NDC for the t = 0 near points
  #     origins + 0 * directions
  origins_ndc = np.stack([
      -2. * fx / width * ox / oz, -2. * fy / height * oy / oz,
      -np.ones_like(oz)
  ],
                         axis=-1)

  # Perspective projection into NDC for the t = infinity far points
  #     origins + infinity * directions
  infinity_ndc = np.stack([
      -2. * fx / width * dx / dz, -2. * fy / height * dy / dz,
      np.ones_like(oz)
  ],
                          axis=-1)

  # directions_ndc points from origins_ndc to infinity_ndc
  directions_ndc = infinity_ndc - origins_ndc

  return origins_ndc, directions_ndc


def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  max_fn = lambda x: max(x, patch_size)
  out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
  img = cv2.resize(img, out_shape, mode)
  return img


def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def shift_origins(origins, directions, near=0.0):
  """Shift ray origins to near plane, such that oz = near."""
  t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions
  return origins


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of near and far bounds in disparity space.
  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
  """Calculates a forward facing spiral path for rendering for DTU."""

  # Get radii for spiral path using 60th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), perc, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  z_axis = focus_pt_fn(poses)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    render_poses.append(viewmatrix(z_axis, up, position, True))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_hemispherical_orbit(poses, n_frames=120):
  """Calculates a render path which orbits around the z-axis."""
  origins = poses[:, :3, 3]
  radius = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))

  # Assume that z-axis points up towards approximate camera hemisphere
  sin_phi = np.mean(origins[:, 2], axis=0) / radius
  cos_phi = np.sqrt(1 - sin_phi**2)
  render_poses = []

  up = np.array([0., 0., 1.])
  for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
    camorigin = radius * np.array(
        [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
    render_poses.append(viewmatrix(camorigin, up, camorigin))

  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def transform_poses_to_hemisphere(poses, bounds):
  """Transforms input poses to lie roughly on the upper unit hemisphere."""

  # Use linear algebra to solve for the nearest point to the set of lines
  # given by each camera's focal axis
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

  # Recenter poses around this point and such that the world space z-axis
  # points up toward the camera hemisphere (based on average camera origin)
  toward_cameras = origins[Ellipsis, 0].mean(0) - focus_pt
  arbitrary_dir = np.array([.1, .2, .3])
  cam2world = viewmatrix(toward_cameras, arbitrary_dir, focus_pt)
  poses_recentered = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  poses_recentered = poses_recentered[Ellipsis, :3, :4]

  # Rescale camera locations (and other metadata) such that average
  # squared distance to the origin is 1 (so cameras lie roughly unit sphere)
  origins = poses_recentered[:, :3, 3]
  avg_distance = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))
  scale_factor = 1. / avg_distance
  poses_recentered[:, :3, 3] *= scale_factor
  bounds_recentered = bounds * scale_factor

  return poses_recentered, bounds_recentered


def subsample_patches(images, patch_size, batch_size, batching='all_images'):
  """Subsamples patches."""
  n_patches = batch_size // (patch_size ** 2)

  scale = np.random.randint(0, len(images))
  images = images[scale]

  if isinstance(images, np.ndarray):
    shape = images.shape
  else:
    shape = images.origins.shape

  # Sample images
  if batching == 'all_images':
    idx_img = np.random.randint(0, shape[0], size=(n_patches, 1))
  elif batching == 'single_image':
    idx_img = np.random.randint(0, shape[0])
    idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
  else:
    raise ValueError('Not supported batching type!')

  # Sample start locations
  x0 = np.random.randint(0, shape[2] - patch_size + 1, size=(n_patches, 1, 1))
  y0 = np.random.randint(0, shape[1] - patch_size + 1, size=(n_patches, 1, 1))
  xy0 = np.concatenate([x0, y0], axis=-1)
  patch_idx = xy0 + np.stack(
      np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
      axis=-1).reshape(1, -1, 2)

  # Subsample images
  if isinstance(images, np.ndarray):
    out = images[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
  else:
    out = utils.dataclass_map(
        lambda x: x[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(  # pylint: disable=g-long-lambda
            -1, x.shape[-1]), images)
  return out, np.ones((n_patches, 1), dtype=np.float32) * scale


def anneal_nearfar(d, it, near_final, far_final,
                   n_steps=2000, init_perc=0.2, mid_perc=0.5):
  """Anneals near and far plane."""
  mid = near_final + mid_perc * (far_final - near_final)

  near_init = mid + init_perc * (near_final - mid)
  far_init = mid + init_perc * (far_final - mid)

  weight = min(it * 1.0 / n_steps, 1.0)

  near_i = near_init + weight * (near_final - near_init)
  far_i = far_init + weight * (far_final - far_init)

  out_dict = {}
  for (k, v) in d.items():
    if 'rays' in k and isinstance(v, utils.Rays):
      ones = np.ones_like(v.origins[Ellipsis, :1])
      rays_out = utils.Rays(
          origins=v.origins, directions=v.directions,
          viewdirs=v.viewdirs, radii=v.radii,
          lossmult=v.lossmult, near=ones*near_i, far=ones*far_i)
      out_dict[k] = rays_out
    else:
      out_dict[k] = v
  return out_dict


def sample_recon_scale(image_list, dist='uniform_scale'):
  """Samples a scale factor for the reconstruction loss."""
  if dist == 'uniform_scale':
    idx = np.random.randint(len(image_list))
  elif dist == 'uniform_size':
    n_img = np.array([i.shape[0] for i in image_list], dtype=np.float32)
    probs = n_img / np.sum(n_img)
    idx = np.random.choice(np.arange(len(image_list)), size=(), p=probs)
  return idx

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
       
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs



class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.use_tiffs = config.use_tiffs
    self.load_disps = config.compute_disp_metrics
    self.load_normals = config.compute_normal_metrics
    self.load_random_rays = config.load_random_rays
    self.load_random_fullimage_rays = config.dietnerf_loss_mult != 0.0
    self.load_masks = ((config.dataset_loader == 'dtu') and (split == 'test')
                       and (not config.dtu_no_mask_eval)
                       and (not config.render_path))
    self.split = split
    self.config = config

    self.data_base_dir = data_dir
    data_dir = os.path.join(data_dir, config.llff_scan)

    self.data_dir = data_dir
    self.near = config.near  ##
    self.far = config.far    ##
    self.near_origin = config.near_origin
    self.anneal_nearfar = config.anneal_nearfar
    self.anneal_nearfar_steps = config.anneal_nearfar_steps
    self.anneal_nearfar_perc = config.anneal_nearfar_perc             ####
    self.anneal_mid_perc = config.anneal_mid_perc                     ####
    self.sample_reconscale_dist = config.sample_reconscale_dist

    if split == 'train':
      self._train_init(config)
    elif split == 'test':
    #   self._train_init(config)
      self._test_init(config)
    else:
      raise ValueError(
          f'`split` should be \'train\' or \'test\', but is \'{split}\'.')
    self.batch_size = config.batch_size // jax.host_count()
    self.batch_size_random = config.batch_size_random // jax.host_count()
    print('Using following batch size', self.batch_size)
    self.patch_size = config.patch_size
    self.batching = config.batching
    self.batching_random = config.batching_random
    self.render_path = config.render_path
    self.render_train = config.render_train
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.get()
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.queue[0].copy()  ## Make a copy of the front of the queue.
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train_mask #
      
    else:     
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_downsampled_images(config)
    self._generate_rays(config)
    self._generate_downsampled_rays(config)
    # Generate more rays / image patches for unobserved-view-based losses.
    if self.load_random_rays:
      self._generate_random_rays(config)
    if self.load_random_fullimage_rays:
      self._generate_random_fullimage_rays(config)
      self._load_renderings_featloss(config)

    self.it = 0
    self.images_noreshape = self.images[0]

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = [i.reshape(-1, 3) for i in self.images]
      if self.load_disps:
        self.disp_images = self.disp_images.flatten()
      if self.load_normals:
        self.normal_images = self.normal_images.reshape([-1, 3])

      self.ray_noreshape = [self.rays]
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, r.shape[-1]]), i) for (i, res) in zip(
              self.rays, self.resolutions)]

    elif config.batching == 'single_image':
      self.images = [i.reshape(
          [-1, r, 3]) for (i, r) in zip(self.images, self.resolutions)]
      if self.load_disps:
        self.disp_images = self.disp_images.reshape([-1, self.resolution])
      if self.load_normals:
        self.normal_images = self.normal_images.reshape(
            [-1, self.resolution, 3])

      self.ray_noreshape = [self.rays]
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, res, r.shape[-1]]), i) for (i, res) in  # pylint: disable=cell-var-from-loop
                   zip(self.rays, self.resolutions)]
    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')

  def _test_init(self, config):
    self._load_renderings(config)
    if self.load_masks:
      self._load_masks(config)
    self._generate_rays(config)
    self.it = 0
  
  def _next_train_mask(self):
    """Sample next training batch."""
    self.it = self.it + 1
    return_dict = {}
    if self.batching == 'all_images':
      # sample scale
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      ray_indices = np.random.randint(0, self.rays[idxs].origins.shape[0], (self.batch_size,))
      return_dict['rgb'] = self.images[idxs][ray_indices]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[ray_indices], self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[ray_indices]
    elif self.batching == 'single_image':
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      image_index = np.random.randint(0, self.n_examples, ())
      if True:
        h, w = self.depth_images[image_index].shape # (3024, 4032)--->(h, w)
        ray_indices = []
        folds = 4
        depth_image = self.depth_images[image_index]        
        max_min_depth = (depth_image.max()-depth_image.min())*1.0

        box_h = 30
        box_w = 30
        mask = (self.depth_images[image_index]>0)

        h_r = np.zeros((self.batch_size//folds//2,2), dtype=np.int32)
        w_r = np.zeros((self.batch_size//folds//2,2), dtype=np.int32)
        i = 0

        while True:
          h_r_ = np.random.randint(h-box_h+1, size=(1, ))[0]
          w_r_ = np.random.randint(w-box_w+1, size=(1, ))[0]          
          # discriminate if boxes contains more than two points
          bbox_mask = mask[h_r_:h_r_+box_h, w_r_:w_r_+box_w]
          if np.sum(bbox_mask)<=10:
            continue
          
          bbox_mask_vec = bbox_mask.reshape(box_h*box_w,)
          mask_ind = np.argwhere(bbox_mask_vec)

          # mask_ind
          selected_ind_ind = np.random.randint(mask_ind.shape[0], size=(2,))
          selected_ind = mask_ind[selected_ind_ind]

          selected_ind_h_ =  selected_ind//box_w    # label0_neighbor_coor_h_
          selected_ind_w_ =  selected_ind-selected_ind_h_*box_w 
          coor_h = selected_ind_h_+h_r_
          coor_w = selected_ind_w_+w_r_

          h_r[i,:] = coor_h.transpose()
          w_r[i,:] = coor_w.transpose()


          i = i +1
          if i>=(self.batch_size//folds//2):
            break

        for j in range(self.batch_size//folds//2): #depth_image 
          label0 = depth_image[h_r[j,0],w_r[j,0]]
          label1 = depth_image[h_r[j,1],w_r[j,1]]
          box_nested_h = 3 
          box_nested_w = 3 
          top_percent = 0.3
          # find the neighbors of label0 
          nest_h_min0 = np.maximum(h_r[j,0]-box_nested_h,0)
          nest_h_max0 = np.minimum(h_r[j,0]+box_nested_h,h-1)

          nest_w_min0 = np.maximum(w_r[j,0]-box_nested_w,0)
          nest_w_max0 = np.minimum(w_r[j,0]+box_nested_w,w-1)
          crop_img_resize0 = depth_image[nest_h_min0:nest_h_max0,nest_w_min0:nest_w_max0]

          # depth2vector = crop_img_resize0[h_s[j,0]-h_r:h_s[j,0]-h_r+box_nested_h, w_s[j,0]-w_r:w_s[j,0]-w_r+box_nested_w].reshape(box_nested_h*box_nested_w,)*1.0
          crop_h, crop_w = crop_img_resize0.shape
          depth2vector0 = crop_img_resize0.reshape(crop_h*crop_w,)*1.0
          # label0
          sorted_ind0 = np.argsort(np.abs(depth2vector0-label0)) # from small to large
          label0_neighbor_ = np.random.randint(int(crop_h*crop_w*top_percent), size=1)[0]
          label0_neighbor = sorted_ind0[label0_neighbor_]

          label0_neighbor_coor_h_ =  label0_neighbor//crop_w
          label0_neighbor_coor_w_ =  label0_neighbor-label0_neighbor_coor_h_*crop_w 
          label0_neighbor_coor_h = label0_neighbor_coor_h_+nest_h_min0
          label0_neighbor_coor_w = label0_neighbor_coor_w_+nest_w_min0

          nest_h_min1 = np.maximum(h_r[j,1]-box_nested_h,0)
          nest_h_max1 = np.minimum(h_r[j,1]+box_nested_h,h-1)

          nest_w_min1 = np.maximum(w_r[j,1]-box_nested_w,0)
          nest_w_max1 = np.minimum(w_r[j,1]+box_nested_w,w-1)

          crop_img_resize1 = depth_image[nest_h_min1:nest_h_max1,nest_w_min1:nest_w_max1]
          crop_h, crop_w = crop_img_resize1.shape
          depth2vector1 = crop_img_resize1.reshape(crop_h*crop_w,)*1.0
          # label0
          sorted_ind1 = np.argsort(np.abs(depth2vector1-label1)) # from small to large
          label1_neighbor_ = np.random.randint(int(crop_h*crop_w*top_percent), size=1)[0]
          label1_neighbor = sorted_ind1[label1_neighbor_]

          label1_neighbor_coor_h_ =  label1_neighbor//crop_w
          label1_neighbor_coor_w_ =  label1_neighbor-label1_neighbor_coor_h_*crop_w
          label1_neighbor_coor_h = label1_neighbor_coor_h_+nest_h_min1
          label1_neighbor_coor_w = label1_neighbor_coor_w_+nest_w_min1
          
          kinect_data = False
          if kinect_data:
            black_is_nearer = (label0<=label1)
          else:
            black_is_nearer = (label0>=label1)

          if black_is_nearer: # white is nearer
            ray_indices.append(h_r[j,0]*w+w_r[j,0])
            ray_indices.append(h_r[j,1]*w+w_r[j,1]) #
            
            ray_indices.append(label0_neighbor_coor_h*w+label0_neighbor_coor_w)
            ray_indices.append(label1_neighbor_coor_h*w+label1_neighbor_coor_w) # 
          else:
            ray_indices.append(h_r[j,1]*w+w_r[j,1])
            ray_indices.append(h_r[j,0]*w+w_r[j,0]) # label0 first
            
            ray_indices.append(label1_neighbor_coor_h*w+label1_neighbor_coor_w) # 
            ray_indices.append(label0_neighbor_coor_h*w+label0_neighbor_coor_w)

        h_r = np.random.randint(h-box_h+1, size=(self.batch_size//2, 1))
        w_r = np.random.randint(w-box_w+1, size=(self.batch_size//2, 1))

        h_r_delta = np.random.randint(box_h, size=(self.batch_size//2, 1))
        w_r_delta = np.random.randint(box_w, size=(self.batch_size//2, 1))

        h_r = h_r+h_r_delta[:,:1] #np.concatenate((h_r+h_r_delta[:,:1],h_r+h_r_delta[:,1:2]), axis=1)
        w_r = w_r+w_r_delta[:,:1] #np.concatenate((w_r+w_r_delta[:,:1],w_r+w_r_delta[:,1:2]), axis=1)          

        for j in range(self.batch_size//2):   # depth_image 
          label0 = depth_image[h_r[j,0],w_r[j,0]]
          ray_indices.append(h_r[j,0]*w+w_r[j,0]) # label0 first
        ray_indices = np.array(ray_indices) 

      else:
        ray_indices = np.random.randint(0, self.rays[idxs].origins[0].shape[0],(self.batch_size,))
           
      return_dict['rgb'] = self.images[idxs][image_index][ray_indices]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[image_index][ray_indices], self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[image_index][ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[image_index][ray_indices]
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')

    if self.load_random_rays:
      return_dict['rays_random'], return_dict['rays_random_scale'] = (
          subsample_patches(self.random_rays, self.patch_size,
                            self.batch_size_random,
                            batching=self.batching_random))
      return_dict['rays_random2'], return_dict['rays_random2_scale'] = (
          subsample_patches(
              self.random_rays, self.patch_size, self.batch_size_random,
              batching=self.batching_random))
    if self.load_random_fullimage_rays:
      idx_img = np.random.randint(self.random_fullimage_rays.origins.shape[0])
      return_dict['rays_feat'] = utils.dataclass_map(
          lambda x: x[idx_img].reshape(-1, x.shape[-1]),
          self.random_fullimage_rays)
      idx_img = np.random.randint(self.images_feat.shape[0])
      return_dict['image_feat'] = self.images_feat[idx_img].reshape(-1, 3)

    if self.anneal_nearfar:
      return_dict = anneal_nearfar(return_dict, self.it, self.near, self.far,
                                   self.anneal_nearfar_steps,
                                   self.anneal_nearfar_perc,
                                   self.anneal_mid_perc)

    return return_dict


  def _next_test(self):
    """Sample next test example."""

    return_dict = {}

    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx],
                                                self.render_rays)
    else:
      return_dict['rgb'] = self.images[idx]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx], self.rays)

    if self.load_masks:
      return_dict['mask'] = self.masks[idx]
    if self.load_disps:
      return_dict['disps'] = self.disp_images[idx]
    if self.load_normals:
      return_dict['normals'] = self.normal_images[idx]

    return return_dict

  def _generate_rays(self, config):
    """Generating rays for all images."""
    del config  # Unused.
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.width, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.height, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')

    camera_dirs = np.stack(
        [(x - self.width * 0.5 + 0.5) / self.focal,
         -(y - self.height * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)

    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))

    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1], directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        lossmult=ones,
        radii=radii,
        near=ones * self.near,
        far=ones * self.far)
    self.render_rays = self.rays


  def _generate_random_poses(self, config):
    """Generates random poses."""
    if config.random_pose_type == 'allposes':
      random_poses = list(self.camtoworlds_all)
    elif config.random_pose_type == 'renderpath':
      def sample_on_sphere(n_samples, only_upper=True, radius=4.03112885717555):
        p = np.random.randn(n_samples, 3)
        if only_upper:
          p[:, -1] = abs(p[:, -1])
        p = p / np.linalg.norm(p, axis=-1, keepdims=True) * radius
        return p

      def create_look_at(eye, target=np.array([0, 0, 0]),
                         up=np.array([0, 0, 1]), dtype=np.float32):
        """Creates lookat matrix."""
        eye = eye.reshape(-1, 3).astype(dtype)
        target = target.reshape(-1, 3).astype(dtype)
        up = up.reshape(-1, 3).astype(dtype)

        def normalize_vec(x, eps=1e-9):
          return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

        forward = normalize_vec(target - eye)
        side = normalize_vec(np.cross(forward, up))
        up = normalize_vec(np.cross(side, forward))

        up = up * np.array([1., 1., 1.]).reshape(-1, 3)
        forward = forward * np.array([-1., -1., -1.]).reshape(-1, 3)

        rot = np.stack([side, up, forward], axis=-1).astype(dtype)
        return rot

      origins = sample_on_sphere(config.n_random_poses)
      rotations = create_look_at(origins)
      random_poses = np.concatenate([rotations, origins[:, :, None]], axis=-1)
    else:
      raise ValueError('Not supported random pose type.')
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generating rays for all images."""
    self._generate_random_poses(config)

    random_rays = []
    for sfactor in [2**i for i in range(config.random_scales_init, config.random_scales)]:
      w = self.width // sfactor
      h = self.height // sfactor
      f = self.focal / (sfactor * 1.0)
      x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32),  # X-Axis (columns)
          np.arange(h, dtype=np.float32),  # Y-Axis (rows)
          indexing='xy')
      
      camera_dirs = np.stack(
          [(x - w * 0.5 + 0.5) / f,
           -(y - h * 0.5 + 0.5) / f, -np.ones_like(x)],
          axis=-1)
      
      directions = ((camera_dirs[None, Ellipsis, None, :] *
                     self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
      
      origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                                directions.shape)
      
      viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
      # Cut the distance in half, multiply it to match the variance of a uniform
      # distribution the size of a pixel (1/12, see paper).
      radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

      ones = np.ones_like(origins[Ellipsis, :1])
      rays = utils.Rays(
          origins=origins,
          directions=directions,
          viewdirs=viewdirs,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
      random_rays.append(rays)
    self.random_rays = random_rays

  def _load_renderings_featloss(self, config):
    """Loades renderings for DietNeRF's feature loss."""
    images = self.images[0]
    res = config.dietnerf_loss_resolution
    images_feat = []
    for img in images:
      images_feat.append(cv2.resize(img, (res, res), cv2.INTER_AREA))
    self.images_feat = np.stack(images_feat)

  def _generate_random_fullimage_rays(self, config):
    """Generating random rays for full images."""
    self._generate_random_poses(config)

    width = config.dietnerf_loss_resolution
    height = config.dietnerf_loss_resolution
    f = self.focal / (self.width * 1.0 / width)

    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(width, dtype=np.float32) + .5,
        np.arange(height, dtype=np.float32) + .5,
        indexing='xy')

    camera_dirs = np.stack([(x - width * 0.5 + 0.5) / f,
                            -(y - height * 0.5 + 0.5) / f,
                            -np.ones_like(x)], axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.random_fullimage_rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)

  def _generate_downsampled_images(self, config):
    """Generating downsampled images."""
    images = []
    resolutions = []
    for sfactor in [2**i for i in range(config.recon_loss_scales)]:
      imgi = np.stack([downsample(i, sfactor) for i in self.images])
      images.append(imgi)
      resolutions.append(imgi.shape[1] * imgi.shape[2])

    self.images = images
    self.resolutions = resolutions

  def _generate_downsampled_rays(self, config):
    """Generating downsampled images."""
    rays, height, width, focal = self.rays, self.height, self.width, self.focal
    ray_list = [rays]
    for sfactor in [2**i for i in range(1, config.recon_loss_scales)]:
      self.height = height // sfactor
      self.width = width // sfactor
      self.focal = focal * 1.0 / sfactor
      self._generate_rays(config)
      ray_list.append(self.rays)
    self.height = height
    self.width = width
    self.focal = focal
    self.rays = ray_list


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    # Load images.
    imgdir_suffix = ''
    if config.factor > 1:
      imgdir_suffix = f'_{config.factor}'
      factor = config.factor
    else:
      factor = 1
    imgdir = os.path.join(self.data_dir, 'images' + imgdir_suffix)
    if not utils.file_exists(imgdir):
      raise ValueError(f'Image folder {imgdir} does not exist.')
    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(utils.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    
    imgdir_org = os.path.join(self.data_dir, 'images')
    self.imgfiles_train = [
        f for f in sorted(utils.listdir(imgdir_org))
        if f.endswith('_train.JPG') or f.endswith('_train.jpg') or f.endswith('_train.png')
    ]    
    basedir = os.path.dirname(imgfiles[0])
    basedir = os.path.dirname(basedir)

    images = []
    depth_images = []
    for imgfile in imgfiles:
      with utils.open_file(imgfile, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        images.append(image)

        image_uint8 = cv2.imread(imgfile)
        h, w, _ = image_uint8.shape

        depth_file = os.path.join(self.config.data_dir, imgfile.split('/')[-3], 'depth_maps', imgfile.split('/')[-1])
        depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) # or -1
        depth_img = cv2.resize(depth_img, (w, h))

        depth_images.append(depth_img)

    images = np.stack(images, axis=0)

    depth_images = np.stack(depth_images, axis=0)

    # Load poses and bounds.
    with utils.open_file(os.path.join(self.data_dir, 'poses_bounds.npy'), 'rb') as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    # import pdb
    # pdb.set_trace()
    bounds = poses_arr[:, -2:]
    if poses.shape[0] != images.shape[0]:
      raise RuntimeError(f'images.shape[0]={images.shape[0]}, ' +
                         f'but poses.shape[0]={poses.shape[0]}')

    # Pull out focal length before processing poses.
    self.focal = poses[0, -1, -1] / factor   ### lwqwgc, problem

    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    poses = poses[:, :3, :4] @ fix_rotation

    # Rescale according to a default bd factor.
    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale


    # Recenter poses.
    poses = recenter_poses(poses)

    # Separate out 360 versus forward facing scenes.
    if config.remap_to_hemisphere: # false
      poses, bounds = transform_poses_to_hemisphere(poses, bounds)
      self.render_poses = generate_hemispherical_orbit(
          poses, n_frames=config.render_path_frames)
    else:
      self.render_poses = generate_spiral_path(
          poses, bounds, n_frames=config.render_path_frames)

    self.use_ndc_space = not config.remap_to_hemisphere

    self.camtoworlds_all = poses
    self.bounds = bounds
    self.images_all = images

    train_indices = []
    test_indices = []
    for i, f in enumerate(imgfiles):
        # assert(f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'))
        if f.endswith('_train.JPG') or f.endswith('_train.jpg') or f.endswith('_train.png'):
            train_indices.append(i)
        else:
            test_indices.append(i)    
    if self.split == "train":
        indices = np.array(train_indices)
    else:
        indices = np.array(test_indices) 

    images = images[indices]
    poses = poses[indices]
    img_file_selected = [imgfiles[ind] for ind in indices]


    depth_images = depth_images[indices]


    self.images = images
    self.depth_images = depth_images
    self.camtoworlds = poses
    self.height, self.width = images.shape[1:3]
    self.resolution = self.height * self.width
    if config.render_path:
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]
    
  def _generate_rays(self, config):
    """Generate normalized device coordinate rays for llff."""
    if self.split == 'test':
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds], axis=0)

    super()._generate_rays(config)

    def adjust_rays_to_ndc(rays, focal, width, height):
      ndc_origins, ndc_directions = convert_to_ndc(rays.origins,rays.directions,focal, width, height)
    #   ndc_origins, ndc_directions = rays.origins, rays.directions# 

      mat = ndc_origins
      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.linalg.norm(mat[:, :-1, :, :] - mat[:, 1:, :, :], axis=-1)
      dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
      dy = np.linalg.norm(mat[:, :, :-1, :] - mat[:, :, 1:, :], axis=-1)
      dy = np.concatenate([dy, dy[:, :, -2:-1]], axis=2)
      # Cut the distance in half, multiply it to match the variance of a uniform
      # distribution the size of a pixel (1/12, see paper).
      radii = (0.5 * (dx + dy))[Ellipsis, None] * 2 / np.sqrt(12)
      ones = np.ones_like(ndc_origins[Ellipsis, :1])
      rays = utils.Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=rays.directions,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
      return rays

    self.rays = adjust_rays_to_ndc(self.rays, self.focal, self.width, self.height) # lwqwgc

    # Split poses from the dataset and generated poses
    if self.split == 'test':
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      self.render_rays, self.rays = (  # Split self.rays into two parts.
          utils.dataclass_map(lambda r: r[:n_render_poses, Ellipsis], self.rays),
          utils.dataclass_map(lambda r: r[n_render_poses:, Ellipsis], self.rays))

  def _generate_downsampled_rays(self, config):
    """Generating downsampled images."""
    rays, height, width, focal = self.rays, self.height, self.width, self.focal
    ray_list = [rays]
    for sfactor in [2**i for i in range(1, config.recon_loss_scales)]:
      self.height = height // sfactor
      self.width = width // sfactor
      self.focal = focal * 1.0 / sfactor
      self._generate_rays(config)
      ray_list.append(self.rays)
    self.height = height
    self.width = width
    self.focal = focal
    self.rays = ray_list

  def _generate_random_poses(self, config):
    """Generates random poses."""
    n_poses = config.n_random_poses
    poses = self.camtoworlds_all
    bounds = self.bounds

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
      t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal, 1.]
      z_axis = position - lookat
    random_poses.append(viewmatrix(z_axis, up, position))
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generates random rays."""
    self._generate_random_poses(config)
    camtoworlds = self.random_poses

    random_rays = []
    for sfactor in [2**i for i in range(config.random_scales_init,
                                        config.random_scales)]:
      w = self.width // sfactor
      h = self.height // sfactor
      f = self.focal / (sfactor * 1.0)

      x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32),  # X-Axis (columns)
          np.arange(h, dtype=np.float32),  # Y-Axis (rows)
          indexing='xy')
      camera_dirs = np.stack(
          [(x - w * 0.5 + 0.5) / f,
           -(y - h * 0.5 + 0.5) / f, -np.ones_like(x)],
          axis=-1)
      directions = ((camera_dirs[None, Ellipsis, None, :] *
                     camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
      origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                                directions.shape)
      viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

      if self.use_ndc_space:
        origins, directions = convert_to_ndc(origins, directions, f, w, h)
      mat = origins
      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.linalg.norm(mat[:, :-1, :, :] - mat[:, 1:, :, :], axis=-1)
      dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
      dy = np.linalg.norm(mat[:, :, :-1, :] - mat[:, :, 1:, :], axis=-1)
      dy = np.concatenate([dy, dy[:, :, -2:-1]], axis=2)
      # Cut the distance in half, multiply it to match the variance of a uniform
      # distribution the size of a pixel (1/12, see paper).
      radii = (0.5 * (dx + dy))[Ellipsis, None] * 2 / np.sqrt(12)
      ones = np.ones_like(origins[Ellipsis, :1])
      random_rays.append(utils.Rays(
          origins=origins,
          directions=directions,
          viewdirs=viewdirs,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far))
    self.random_rays = random_rays

  def _generate_random_fullimage_rays(self, config):
    """Generates random full image rays for DietNeRF loss."""
    self._generate_random_poses(config)
    camtoworlds = self.random_poses
    w = config.dietnerf_loss_resolution
    h = config.dietnerf_loss_resolution
    fx = self.focal / (self.width * 1.0 / w)
    fy = self.focal / (self.height * 1.0 / h)

    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - w * 0.5 + 0.5) / fx,
         -(y - h * 0.5 + 0.5) / fy, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    if self.use_ndc_space:
    # if False:
      origins, directions = convert_to_ndc(origins, directions, fx, w, h,
                                           focaly=fy)
    mat = origins
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.linalg.norm(mat[:, :-1, :, :] - mat[:, 1:, :, :], axis=-1)
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    dy = np.linalg.norm(mat[:, :, :-1, :] - mat[:, :, 1:, :], axis=-1)
    dy = np.concatenate([dy, dy[:, :, -2:-1]], axis=2)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = (0.5 * (dx + dy))[Ellipsis, None] * 2 / np.sqrt(12)
    ones = np.ones_like(origins[Ellipsis, :1])
    self.random_fullimage_rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)

