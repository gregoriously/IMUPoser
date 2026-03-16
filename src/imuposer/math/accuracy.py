import torch

from imuposer.math.angular import r6d_to_rotation_matrix


def calc_mpjpe(target, pred, bodymodel):
    def FK(pose):
        return bodymodel.forward_kinematics(
            pose=r6d_to_rotation_matrix(pose).view(-1, 216)
        )[1]

    target_joints = FK(target)
    pred_joints = FK(pred)
    # align by pelvis
    pred_joints = pred_joints - pred_joints[:, 0:1, :]  # pelvis = joint 0
    target_joints = target_joints - target_joints[:, 0:1, :]

    mpjpe = torch.norm(pred_joints - target_joints, dim=-1).mean()

    return mpjpe


def calc_mpjre(target, pred):
    R_pred = r6d_to_rotation_matrix(pred)
    R_target = r6d_to_rotation_matrix(target)
    R_diff = R_pred @ R_target.transpose(-1, -2)
    angle = torch.acos(
        torch.clamp((R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)
    )
    mpjre = angle.mean() * (180 / torch.pi)
    return mpjre


def calc_mpjve():
    pass
