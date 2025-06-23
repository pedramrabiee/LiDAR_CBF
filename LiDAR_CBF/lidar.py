import torch
import matplotlib.pyplot as plt


class Lidar:
    def __init__(self, map, lidar_cfg, dynamics):
        self.map = map
        self.lidar_cfg = lidar_cfg
        self.dynamics = dynamics
        self.sensor_location = torch.tensor(lidar_cfg.sensor_mounting_location)
        self.sensor_angle = torch.tensor(lidar_cfg.sensor_mounting_angle)
        self._make_mesh()

    def update(self, x):
        pos = self.dynamics.get_pos(x)
        rot = self.dynamics.get_rot(x)

        # Repeat global mesh for each batch item
        vectorized_global_mesh = self.global_mesh.repeat(x.shape[0], 1)

        # Transform the mesh from robot to world coordinates
        transformed_vectorized_global_mesh = self._robot_to_world(vectorized_global_mesh, pos, rot)

        boundary_points, detected_points = self._boundary_extractor(transformed_vectorized_global_mesh)

        if self.lidar_cfg.has_noise:
            points = self._add_noise(boundary_points, detected_points)
        return {'boundary_points': boundary_points, 'detected_points': detected_points}

    def _make_mesh(self):
        range_space = torch.linspace(0, self.lidar_cfg.max_range, self.lidar_cfg.ray_sampling_rate, dtype=torch.float64)
        theta_space = torch.linspace(self.lidar_cfg.scan_angle[0], self.lidar_cfg.scan_angle[-1],
                                     self.lidar_cfg.ray_num, dtype=torch.float64)
        r, theta = torch.meshgrid(range_space, theta_space, indexing='xy')
        self.global_mesh = self._sensor_to_robot(self._polar_to_cartesian(r.flatten(), theta.flatten()))


    def _polar_to_cartesian(self, r, theta):
        # (r,theta) --> (x,y)
        return torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)

    def _robot_to_world(self, points, pos, rot):
        batch_size = pos.shape[0]
        # Pre-compute sine and cosine
        c, s = torch.cos(rot), torch.sin(rot)

        # Create rotation matrix
        R = torch.stack([c, -s, s, c], dim=1).view(batch_size, 2, 2)

        # Apply rotation and translation
        rotated = torch.bmm(points.view(batch_size, -1, self.lidar_cfg.space_dimension), R.transpose(1, 2))
        transformed = rotated + pos.unsqueeze(1).repeat(1, rotated.shape[1], 1)

        return transformed.view(-1, self.lidar_cfg.space_dimension)

    def _sensor_to_robot(self, points):
        # 2D transformation from sensor coordinate to robot coordinate
        c, s = torch.cos(self.sensor_angle), torch.sin(self.sensor_angle)

        x = c * points[..., 0] - s * points[..., 1] + self.sensor_location[0]
        y = s * points[..., 0] + c * points[..., 1] + self.sensor_location[1]

        return torch.stack((x, y), dim=-1)


    def _boundary_extractor(self, mesh):
        # Calculate barrier values
        z = self.map.map_barrier.min_barrier(mesh)

        # indices = torch.where(torch.abs(z) < 0.01)[0]
        # neg_ind = torch.where(z < 0)[0]
        # pos_ind = torch.where(z >= 0.0)[0]
        # points_close_to_zero = mesh[indices]
        # pos_points = mesh[pos_ind]
        # neg_points = mesh[neg_ind]
        #
        # xx = torch.linspace(0 - 0.5, 15 + 0.5, 150)
        # yy = torch.linspace(0 - 0.5, 15 + 0.5, 150)
        # X, Y = torch.meshgrid(xx, yy)
        # points = torch.column_stack((X.flatten(), Y.flatten()))
        # points = torch.column_stack((points, torch.zeros(points.shape)))
        # Z = self.map.map_barrier.min_barrier(points).reshape(X.shape)
        # plt.figure(figsize=(8, 6), dpi=100)
        # plt.contour(X, Y, Z, levels=[0], colors='k')
        # # plt.scatter(mesh[:, 0], mesh[:, 1], s=1)
        # plt.scatter(pos_points[:, 0], pos_points[:, 1], c='blue', s = 1)
        # plt.scatter(neg_points[:, 0], neg_points[:, 1], c='red', s=1)
        # # plt.scatter(points_close_to_zero[:, 0], points_close_to_zero[:, 1], s=1, color='red')
        # plt.axis('equal')
        # plt.show()

        # Reshape z and mesh
        batch_size = mesh.shape[0] // (self.lidar_cfg.ray_num * self.lidar_cfg.ray_sampling_rate)
        reshaped_z = z.view(batch_size, self.lidar_cfg.ray_num, self.lidar_cfg.ray_sampling_rate)
        reshaped_mesh = mesh.view(batch_size, self.lidar_cfg.ray_num, self.lidar_cfg.ray_sampling_rate,
                                  self.lidar_cfg.space_dimension)

        # Find boundary_points points
        sign_change = (reshaped_z[:, :, :-1] >=0) & (reshaped_z[:, :, 1:] < 0)
        change_indices = torch.nonzero(sign_change, as_tuple=True)

        boundary_points = reshaped_mesh[:, :, -1, :]
        detected_points = torch.full(boundary_points.shape, float('nan'), dtype=torch.float64)

        if change_indices[0].numel() > 0:
            # Find unique ray indices
            change_indices = torch.stack(change_indices)
            # Create a tuple of the first two rows
            key_rows = tuple(change_indices[:2])

            # Use a dictionary to keep track of first occurrences
            seen = {}
            keep_indices = []

            for i in range(change_indices.shape[1]):
                key = (key_rows[0][i].item(), key_rows[1][i].item())
                if key not in seen:
                    seen[key] = True
                    keep_indices.append(i)

            # Use the indices to select the unique columns
            filtered = change_indices[:, keep_indices]

            # Assign boundary_points points
            detected_points[filtered[0], filtered[1]] = reshaped_mesh[
                filtered[0], filtered[1], filtered[2]]

        # xx = torch.linspace(0 - 0.5, 15 + 0.5, 150)
        # yy = torch.linspace(0 - 0.5, 15 + 0.5, 150)
        # X, Y = torch.meshgrid(xx, yy)
        # points = torch.column_stack((X.flatten(), Y.flatten()))
        # points = torch.column_stack((points, torch.zeros(points.shape)))
        # Z = self.map.map_barrier.min_barrier(points).reshape(X.shape)
        # plt.figure(figsize=(8, 6), dpi=100)
        # plt.contour(X, Y, Z, levels=[0], colors='k')
        # plt.scatter(detected_points[0, :, 0], detected_points[0, :, 1], s=1)
        # plt.axis('equal')
        # plt.show()
        return boundary_points, detected_points


    def _add_noise(self, points, maxrange):
        # TODO : Implement LiDAR Noise
        raise NotImplementedError
