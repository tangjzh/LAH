import numpy as np
import torch
import matplotlib.pyplot as plt

from math import pi, cos, sin

def create_camera_to_world_matrix(elevation, azimuth, face_target=True):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    y = np.cos(elevation) * np.cos(azimuth)
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    # Construct view matrix
    forward = target - camera_pos if face_target else camera_pos - target
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    forward = -forward if face_target else forward
    cam2world[:3, :3] = np.array([right, new_up, forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender


def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix.reshape(-1,16)


def get_camera(num_frames, elevation=15, azimuth_start=0, azimuth_span=360, blender_coord=True):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, 0)).float()

def transform_pose(reference_pose, pose, align_matrix=None):
    reference_R = reference_pose[:3, :3]
    reference_T = reference_pose[:3, 3]
    R = pose[:3, :3]
    T = pose[:3, 3]
    transformed_R = R
    transformed_T = T - reference_T
    transformed_pose = np.hstack((transformed_R, transformed_T[:, np.newaxis]))
    transformed_pose = np.vstack((transformed_pose, [0, 0, 0, 1]))
    return transformed_pose

def generate_poses(angles, elevations, anchor=None):
    poses = []
    
    # Generate poses for different elevations
    for elevation in elevations:
        for azimuth in angles:
            pose = create_camera_to_world_matrix(elevation, azimuth, False)
            poses.append(pose)

    if anchor is None:
        return poses
        
    # Use one of the level poses as the reference camera_matrix
    reference_pose = poses[anchor]
    
    # Adjust all poses based on the reference
    adjusted_poses = np.stack([transform_pose(reference_pose, pose) for pose in poses])
    
    return adjusted_poses

def visualize_extrinsics(extrinsics, save_path=None, length=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='r', s=100, label='Target Point')
    for i, pose in enumerate(extrinsics):
        camera_pos = pose[:3, 3]
        forward = pose[:3, 2]  # Forward vector is the third column
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], label=f'Pose {i+1}')
        ax.text(camera_pos[0], camera_pos[1], camera_pos[2], f'{i+1}', color='blue')
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], 
                  forward[0], forward[1], forward[2], length=length, color='k')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # ax.legend()
    plt.show()
    if save_path:
        plt.savefig(save_path)

def get_extrinsic(camera_pos, rays=None, track_point=None, fourxfour=True):
    """ Returns extrinsic matrix mapping world to camera coordinates.
    Args:
        camera_pos (np array [3]): Camera position.
        track_point (np array [3]): Point on which the camera is fixated.
        rays (np array [h, w, 3]): Rays eminating from the camera. Used to determine track_point
            if it's not given.
        fourxfour (bool): If true, a 4x4 matrix for homogeneous 3D coordinates is returned.
            Otherwise, a 3x4 matrix is returned.
    Returns:
        extrinsic camera matrix (np array [4, 4] or [3, 4])
    """
    if track_point is None:
        h, w, _ = rays.shape
        if h % 2 == 0:
            center_rays = rays[h//2 - 1:h//2 + 1]
        else:
            center_rays = rays[h//2:h//2+1]

        if w % 2 == 0:
            center_rays = rays[:, w//2 - 1:w//2 + 1]
        else:
            center_rays = rays[:, w//2:w//2+1]

        camera_z = center_rays.mean((0, 1))
    else:
        camera_z = track_point - camera_pos

    camera_z = camera_z / np.linalg.norm(camera_z, axis=-1, keepdims=True)

    # We assume that (a) the z-axis is vertical, and that
    # (b) the camera's horizontal, the x-axis, is orthogonal to the vertical, i.e.,
    # the camera is in a level position.
    vertical = np.array((0., 0., 1.))

    camera_x = np.cross(camera_z, vertical)
    camera_x = camera_x / np.linalg.norm(camera_x, axis=-1, keepdims=True)
    camera_y = np.cross(camera_z, camera_x)

    camera_matrix = np.stack((camera_x, camera_y, camera_z), -2)
    translation = -np.einsum('...ij,...j->...i', camera_matrix, camera_pos)
    camera_matrix = np.concatenate((camera_matrix, np.expand_dims(translation, -1)), -1)

    if fourxfour:
        filler = np.array([[0., 0., 0., 1.]])
        camera_matrix = np.concatenate((camera_matrix, filler), 0)
    return camera_matrix


def get_extrinsic_torch(camera_pos, rays=None, track_point=None, fourxfour=True):
    """ Returns extrinsic matrix mapping world to camera coordinates, by mapping the inputs
    to Numpy arrays, calling get_extrinsic above, and mapping back to PyTorch.
    Does not support gradient computation!
    """
    camera_pos_torch = camera_pos
    camera_pos = camera_pos.detach().cpu().numpy()
    if rays is not None:
        rays = rays.detach().cpu().numpy()

    if track_point is not None:
        track_point = track_point.detach().cpu().numpy()

    camera_matrix = get_extrinsic(camera_pos, rays, track_point, fourxfour=fourxfour)
    return torch.Tensor(camera_matrix).to(camera_pos_torch)


def transform_points(points, transform, translate=True):
    """ Apply linear transform to a np array of points.
    Args:
        points (np array [..., 3]): Points to transform.
        transform (np array [3, 4] or [4, 4]): Linear map.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (np array [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum('nm,...m->...n', transform, points)
    return points[..., :3]


def transform_points_torch(points, transform, translate=True):
    """ Apply a batch of linear transforms to a PyTorch tensor of points.
    Args:
        points (Tensor [..., 3]): Points to transform.
        transform (Tensor [..., 3, 4] or [..., 4, 4]): Linear maps.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (Tensor [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = torch.ones_like(points[..., :1])
    else:
        constant_term = torch.zeros_like(points[..., :1])
    points = torch.cat((points, constant_term), dim=-1)

    points = torch.einsum('...nm,...m->...n', transform, points)
    return points[..., :3]


def rotate_around_z_axis_np(points, theta):
    # Rotate point around the z axis
    results = np.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = cos(theta) * points[..., 0] - sin(theta) * points[..., 1]
    results[..., 1] = sin(theta) * points[..., 0] + cos(theta) * points[..., 1]

    return results


def rotate_around_z_axis_torch(points, theta):
    # Rotate point around the z axis
    results = torch.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = cos(theta) * points[..., 0] - sin(theta) * points[..., 1]
    results[..., 1] = sin(theta) * points[..., 0] + cos(theta) * points[..., 1]
    return results

def generate_rays_with_extrinsics(extrinsic_matrix, width=320, height=240, focal_length=0.035, sensor_width=0.032, noisy=False, vertical=None):
    """
    Generate unit direction vectors (rays) from the camera position to each pixel on the image plane using extrinsic matrix.
    
    Args:
    - extrinsic_matrix (np.array): 4x4 camera extrinsic matrix [R|T].
    - width (int): Image width in pixels.
    - height (int): Image height in pixels.
    - focal_length (float): Focal length of the camera.
    - sensor_width (float): Width of the sensor.
    - noisy (bool): Whether to add noise to the pixel positions.
    - vertical (np.array): The vertical direction vector.
    
    Returns:
    - rays (np.array): Array of unit direction vectors, shape (height, width, 3).
    """
    # Camera position from the extrinsic matrix
    c_pos = extrinsic_matrix[:3, 3]

    # The camera direction vector (negative Z axis in camera coordinate system)
    c_dir = -extrinsic_matrix[:3, 2]
    c_dir = c_dir / np.linalg.norm(c_dir)

    # Center of the image plane in world coordinates
    img_plane_center = c_pos + c_dir * focal_length

    # The horizontal axis of the camera sensor is horizontal (z=0) and orthogonal to the view axis
    if vertical is None:
        vertical = np.array([0.0, 1.0, 0.0])
    img_plane_horizontal = np.cross(c_dir, vertical)
    img_plane_horizontal = img_plane_horizontal / np.linalg.norm(img_plane_horizontal)

    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

    # Double check that everything is orthogonal
    def is_small(x, atol=1e-7):
        return abs(x) < atol

    assert is_small(np.dot(img_plane_vertical, img_plane_horizontal))
    assert is_small(np.dot(img_plane_vertical, c_dir))
    assert is_small(np.dot(c_dir, img_plane_horizontal))

    # Sensor height is implied by sensor width and aspect ratio
    sensor_height = (sensor_width / width) * height

    # Compute pixel boundaries
    horizontal_offsets = np.linspace(-1, 1, width+1) * sensor_width / 2
    vertical_offsets = np.linspace(-1, 1, height+1) * sensor_height / 2

    # Compute pixel centers
    horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
    vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

    horizontal_offsets = np.repeat(np.reshape(horizontal_offsets, (1, width)), height, 0)
    vertical_offsets = np.repeat(np.reshape(vertical_offsets, (height, 1)), width, 1)

    if noisy:
        pixel_width = sensor_width / width
        pixel_height = sensor_height / height
        horizontal_offsets += (np.random.random((height, width)) - 0.5) * pixel_width
        vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

    horizontal_offsets = (np.reshape(horizontal_offsets, (height, width, 1)) *
                          np.reshape(img_plane_horizontal, (1, 1, 3)))
    vertical_offsets = (np.reshape(vertical_offsets, (height, width, 1)) *
                        np.reshape(img_plane_vertical, (1, 1, 3)))

    image_plane = horizontal_offsets + vertical_offsets
    image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
    c_pos_exp = np.reshape(c_pos, (1, 1, 3))
    rays = image_plane - c_pos_exp
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / ray_norms
    return rays.astype(np.float32)

def get_camera_rays(c_pos, width=320, height=240, focal_length=0.035, sensor_width=0.032, noisy=False,
                    vertical=None, track_point=None):
    # The camera is pointed at the origin
    if track_point is None:
        track_point = np.array((0., 0., 0.))

    if vertical is None:
        vertical = np.array((0., 0., 1.))

    c_dir = (track_point - c_pos)
    c_dir = c_dir / np.linalg.norm(c_dir)

    img_plane_center = c_pos + c_dir * focal_length

    # The horizontal axis of the camera sensor is horizontal (z=0) and orthogonal to the view axis
    img_plane_horizontal = np.cross(c_dir, vertical)
    #img_plane_horizontal = np.array((-c_dir[1]/c_dir[0], 1., 0.))
    img_plane_horizontal = img_plane_horizontal / np.linalg.norm(img_plane_horizontal)

    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

    # Double check that everything is orthogonal
    def is_small(x, atol=1e-7):
        return abs(x) < atol

    assert(is_small(np.dot(img_plane_vertical, img_plane_horizontal)))
    assert(is_small(np.dot(img_plane_vertical, c_dir)))
    assert(is_small(np.dot(c_dir, img_plane_horizontal)))

    # Sensor height is implied by sensor width and aspect ratio
    sensor_height = (sensor_width / width) * height

    # Compute pixel boundaries
    horizontal_offsets = np.linspace(-1, 1, width+1) * sensor_width / 2
    vertical_offsets = np.linspace(-1, 1, height+1) * sensor_height / 2

    # Compute pixel centers
    horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
    vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

    horizontal_offsets = np.repeat(np.reshape(horizontal_offsets, (1, width)), height, 0)
    vertical_offsets = np.repeat(np.reshape(vertical_offsets, (height, 1)), width, 1)

    if noisy:
        pixel_width = sensor_width / width
        pixel_height = sensor_height / height
        horizontal_offsets += (np.random.random((height, width)) - 0.5) * pixel_width
        vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

    horizontal_offsets = (np.reshape(horizontal_offsets, (height, width, 1)) *
                          np.reshape(img_plane_horizontal, (1, 1, 3)))
    vertical_offsets = (np.reshape(vertical_offsets, (height, width, 1)) *
                        np.reshape(img_plane_vertical, (1, 1, 3)))

    image_plane = horizontal_offsets + vertical_offsets

    image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
    c_pos_exp = np.reshape(c_pos, (1, 1, 3))
    rays = image_plane - c_pos_exp
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / ray_norms
    return rays.astype(np.float32)


def get_nerf_sample_points(camera_pos, rays, min_dist=0.035, max_dist=30, num_samples=256,
                           min_z=None, mip=False, deterministic=False):
    """
    Get uniform points for coarse NeRF sampling:

    Args:
        camera_pos: [..., 3] tensor indicating camera position
        rays: [..., 3] tensor indicating unit length directions of the pixel rays
        min_dist: focal length of the camera
        max_dist: maximum render distance
        num_samples: number of samples to generate
        mip: Generate mipnerf coordinates
    Return:
        sample_depths: Depths of the sampled points, tensor of shape [..., num_samples]
        sample_points: 3d coordiantes of the sampled points, tensor of shape [..., num_samples, 3]
    """
    max_dist = torch.zeros_like(rays[..., 0]) + max_dist

    if min_z is not None:
        delta_z = min_z - camera_pos[..., 2]
        t_int = delta_z / rays[..., 2]
        t_int_clip = torch.logical_and(t_int >= 0., t_int <= max_dist)

        max_dist[t_int_clip] = t_int[t_int_clip]

    sample_segment_borders = torch.linspace(0., 1., num_samples+1).to(rays)
    while len(sample_segment_borders.shape) <= len(max_dist.shape):
        sample_segment_borders = sample_segment_borders.unsqueeze(0)
    sample_segment_borders = sample_segment_borders * (max_dist - min_dist).unsqueeze(-1) + min_dist

    if deterministic:
        sample_depths = (sample_segment_borders[..., 1:] + sample_segment_borders[..., :-1]) / 2
    else:
        sample_depth_dist = torch.distributions.Uniform(sample_segment_borders[..., :-1],
                                                        sample_segment_borders[..., 1:])
        sample_depths = sample_depth_dist.rsample()

    if mip:
        sample_t0 = sample_segment_borders[:-1]
        sample_t1 = sample_segment_borders[1:]
        base_radii = get_base_radius_torch(rays).flatten(0, 1).unsqueeze(0).repeat(sample_t0.shape[0], 1)
        sample_points = torch.stack((sample_t0, sample_t1, base_radii), -1)
    else:
        scaled_rays = rays.unsqueeze(-2) * sample_depths.unsqueeze(-1)
        sample_points = scaled_rays + camera_pos.unsqueeze(-2)
    return sample_depths, sample_points


def get_fine_nerf_sample_points(camera_pos, rays, depth_dist, depths,
                                min_dist=0.035, max_dist=30., num_samples=256,
                                deterministic=False):
    """
    Get points for fine NeRF sampling:

    Args:
        camera_pos: [..., 3] tensor indicating camera position
        rays: [..., 3] tensor indicating unit length directions of the pixel rays
        depth_dist: [..., s] tensor indicating the depth distribution obtained so far.
            Must sum to one along the s axis.
        depths: [..., s] tensor indicating the depths to which the depth distribution is referring.
        min_dist: focal length of the camera
        max_dist: maximum render distance
        num_samples: number of samples to generate
    Return:
        sample_depths: Depths of the sampled points, tensor of shape [..., s]
        sample_points: 3d coordiantes of the sampled points, tensor of shape [..., s, 3]
    """

    segment_borders = torch.cat((torch.zeros_like(depths[..., :1]) + min_dist,
                                 depths,
                                 1.5 * depths[..., -1:] - 0.5 * depths[..., -2:-1]), -1)
    histogram_weights = torch.zeros_like(segment_borders[..., 1:])

    # Allocate 75% of samples to previous segment, 0.25 to the following one.
    histogram_weights[..., :-1] = depth_dist * 0.75
    histogram_weights[..., 1:] += depth_dist * 0.25

    sample_depths = sample_pdf(segment_borders, histogram_weights, num_samples, deterministic=deterministic)
    scaled_rays = rays.unsqueeze(-2) * sample_depths.unsqueeze(-1)

    sample_points = scaled_rays + camera_pos.unsqueeze(-2)
    return sample_depths, sample_points


def sample_pdf(bins, depth_dist, num_samples, deterministic=False):
    """ Sample from histogram. Adapted from github.com/bmild/nerf
    Args:
        bins: Boundaries of the histogram bins, shape [..., s+1]
        depth_dist: Probability of each bin, shape [..., s]. Must sum to one along the s axis.
        num_samples: Number of samples to collect from each histogram.
        deterministic: Whether to collect linearly spaced samples instead of sampling.
    """

    # Get pdf
    depth_dist += 1e-5  # prevent nans
    cdf = torch.cumsum(depth_dist, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1).contiguous()

    sample_shape = list(bins.shape[:-1]) + [num_samples]

    # Take uniform samples
    if deterministic:
        u = torch.linspace(0., 1., num_samples).to(bins)
        u = u.expand(sample_shape).contiguous()
    else:
        u = torch.rand(sample_shape).to(bins)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)

    inds_below = torch.maximum(torch.zeros_like(inds), inds-1)
    inds_above = torch.minimum(torch.zeros_like(inds) + (cdf.shape[-1]-1), inds)

    cdf_below = torch.gather(cdf, -1, inds_below)
    cdf_above = torch.gather(cdf, -1, inds_above)
    bins_below = torch.gather(bins, -1, inds_below)
    bins_above = torch.gather(bins, -1, inds_above)

    denom = (cdf_above - cdf_below)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    return samples


def draw_nerf(pres, values, depths):
    """
    Do the NeRF integration based on the given samples.
    Args:
        Batch dimension is optional.
        pres: Densities of the samples, shape [n, p, s]
        values: Color value of each samples, shape [n, p, s, 3]
        depths: Depth of each sample, shape [n, p, s]

    Returns:
        Batch dimension is optional.
        image: The expected colors of each ray/pixel, [n, p, 3]
        expected_depth: The expected depth of each ray/pixel, [n, p]
        depth_dist: Categorical distribution over the samples, [n, p, s]
    """
    num_points, num_samples, _ = values.shape[-3:]

    if pres.isnan().any():
        print(pres)
        print('pres nan')

    if values.isnan().any():
        print('values nan')

    segment_sizes = depths[..., 1:] - depths[..., :-1]

    last_segment = torch.ones_like(segment_sizes[..., -1:]) * 1e10
    segment_sizes_ext = torch.cat((segment_sizes, last_segment), -1)
    # Log prob that the segment between samples i and i+1 is empty
    # Attributing the density of sample i to that segment.
    prob_segment_empty = torch.exp(-pres * segment_sizes_ext)
    alpha = 1. - prob_segment_empty
    # Log prob that Everything up until segment i+1 is empty
    prob_ray_empty = (prob_segment_empty + 1e-10).cumprod(-1)

    # Prepend 0 to get the log prob that everything until segment i is empty
    prob_ray_empty_shifted = torch.cat((torch.ones_like(prob_ray_empty[..., :1]),
                                        prob_ray_empty[..., :-1]), -1)

    if torch.isnan(alpha).any():
        print('alpha nan')
    segment_probs = alpha * prob_ray_empty_shifted

    total_prob = segment_probs.sum(-1)
    if torch.isnan(total_prob).any():
        print('total density nan')
    bg_prob = prob_ray_empty[..., -1]
    total_alpha = 1. - bg_prob

    expected_values = (values * segment_probs.unsqueeze(-1)).sum(-2)
    expected_depth = (segment_probs * depths).sum(-1)

    image = torch.cat((expected_values, total_alpha.unsqueeze(-1)), -1)

    return image, expected_depth, segment_probs
