pointclouds = []
        pointclouds_seg = []
        for i in range(30):
            angle = 2 * np.pi * i / 30
            rotate = np.array([np.sin(angle), np.cos(angle), -0.1])
            location = np.array([self.agent_pos[0], self.agent_pos[1], 0.05])
            viewMatrix = p.computeViewMatrix(location, location + rotate, [0, 0, 1])
            near = 0.0001
            far = 4.0
            projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)
            _, _, im, three_d, seg = p.getCameraImage(width=128, height=128, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
            seg_ids = np.unique(seg)
            mask = np.zeros((128, 128), dtype=np.bool)
            for seg_id in seg_ids:
                if seg_id in self.obj2idx.keys():
                    seg_mask = (seg == seg_id)
                    mask = mask | seg_mask
            depth = far * near / (far  - (far - near) * three_d)
            viewMatrix = np.array(viewMatrix).reshape((4, 4)).transpose()
            cam2world = np.linalg.inv(viewMatrix)
            focal = 0.5 * 128 / np.tan(0.5 * np.pi / 3)
            rays_o, rays_d = get_rays_np(128, 128, focal, cam2world)
            pointcloud = depth[:, :, None] * rays_d + rays_o
            pointcloud_seg = np.concatenate([pointcloud, seg[:, :, None]], axis=2)
            pointcloud_dense = pointcloud.reshape((-1, 3))
            pointcloud_dense = pointcloud_dense[mask.flatten()]
            pointclouds.append(pointcloud_dense)
            pointclouds_seg.append(pointcloud_seg.reshape((-1, 4)))
            depths.append(depth)
            writer.append_data(im)
            seg_writer.append_data(seg)

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing=‘xy’)
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame’s origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
