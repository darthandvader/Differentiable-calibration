import torch as th


def lndFK(joint_angles: th.Tensor):
    """
    Forward kinematics of LND starting from Frame 4
    param joint_angles: joint angles at Joint 5, 6, 7, 8 (PyTorch tensor)
    return: rotation matrices at Frame 4, 5, 7, 8; (4, 3, 3) PyTorch tensors
    return: translation vectors at Frame 4, 5, 7, 8; (4, 3) PyTorch tensors
    """
    device = joint_angles.device
    dtype = joint_angles.dtype

    # Clone joint angles to avoid in-place modifications on views
    theta0 = joint_angles[0].clone()
    theta1 = joint_angles[1].clone()
    theta2 = joint_angles[2].clone()
    theta3 = joint_angles[3].clone()

    # Frame 4 (Base)
    T_4 = th.eye(
        4, device=device, dtype=dtype
    )  # Identity matrix since we start at Frame 4

    # Transformation from Frame 4 to Frame 5
    T_4_5 = th.stack(
        [
            th.stack(
                [
                    th.sin(theta0),
                    th.cos(theta0),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.cos(theta0),
                    -th.sin(theta0),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_5_6 = th.stack(
        [
            th.stack(
                [
                    th.sin(theta1),
                    th.cos(theta1),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0091, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.cos(theta1),
                    -th.sin(theta1),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    # Mesh transformation matrices (static, not dependent on joint angles)
    T_4_mesh = th.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    T_5_mesh = th.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    # Compute transformations from Frame 4 onward
    T_5 = T_4 @ T_4_5
    T_6 = T_5 @ T_5_6
    T_4 = T_4 @ T_4_mesh
    T_5 = T_5 @ T_5_mesh

    # Frame 7: right gripper
    T_6_7 = th.stack(
        [
            th.stack(
                [
                    th.cos(theta2),
                    th.sin(theta2),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    -th.sin(theta2),
                    th.cos(theta2),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_7_mesh = th.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    T_7 = T_6 @ T_6_7 @ T_7_mesh

    # Frame 8: left gripper
    T_6_8 = th.stack(
        [
            th.stack(
                [
                    th.cos(theta3),
                    -th.sin(theta3),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.sin(theta3),
                    th.cos(theta3),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_8 = T_6 @ T_6_8 @ T_7_mesh

    # Extract rotation matrices and translation vectors

    R_list = th.stack([T_4[:3, :3], T_5[:3, :3], T_7[:3, :3], T_8[:3, :3]], dim=0)
    t_list = th.stack([T_4[:3, 3], T_5[:3, 3], T_7[:3, 3], T_8[:3, 3]], dim=0)

    return R_list, t_list
