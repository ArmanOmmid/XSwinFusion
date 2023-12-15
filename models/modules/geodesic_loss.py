
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class SpecialEuclideanGeodesicLoss(_Loss):
    def __init__(self, PCD_Loss=True, SO_Loss=False, PCD_weight=1.0, SO_weight=0.1) -> None:
        super().__init__()

        self.PCD_Loss = PCD_Loss
        self.PCD_weight = PCD_weight
        if self.PCD_Loss:
            self.PCD_criterion = PointCloudMSELoss(target_type="pose", weight=PCD_weight)
        
        self.SO_Loss = SO_Loss
        if self.SO_Loss:
            self.SO_criterion = SpecialOrthogonalLoss(weight=SO_weight)

    def normalize(self, rot_matrix):
        u, s, v = torch.svd(rot_matrix)
        return torch.bmm(u, v.transpose(-2, -1))

    def forward(self, predicted_transform, target_transform, 
                source_pcd=None, symmetries=None, extra_SO=None, components=True):
        
        # Transforms are 3x4 with a 3x3 in SO(3) and a 3x1 in R(3)
        losses = []

        p_T = predicted_transform[:, :3, 3]
        t_T = target_transform[:, :3, 3]
        translation_loss = F.mse_loss(p_T, t_T)

        p_R = predicted_transform[:, :3, :3]
        t_R = target_transform[:, :3, :3]
        if symmetries is None:
            relative_rotation = torch.bmm(p_R, t_R.transpose(-2, -1))
            batch_trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(dim=-1)
            cos_theta = (batch_trace - 1.0) / 2.0
            cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)  # Numerical stability
            theta = torch.acos(cos_theta)
            rotation_loss = torch.mean(theta) # Over Batch
        else:
            # symmetries # B N 3 3
            p_R_sym = p_R.unsqueeze(1) # B 1 3 3 #predicted
            t_R_sym = t_R.unsqueeze(1) # B 1 3 3 #truth
            
            # Avoid gradient computations by applying to ground truth
            # Compose Transforms (No transpose). Optionally, do tranpose now and avoid it later
            t_R_sym = torch.matmul(symmetries, t_R_sym)
            relative_rotation = torch.matmul(p_R_sym, t_R_sym.transpose(-2, -1))
            cosine_term = (torch.einsum('bnii->bn', relative_rotation) - 1) / 2 # EinSum Trace
            cosine_term = torch.clamp(cosine_term, -0.9999, 0.9999)  # Numerical stability
            symmetry_losses = torch.acos(cosine_term)  # Shape: (B, N)
            # Get minimum loss and compute mean
            argmin_symmetry = torch.argmin(symmetry_losses, dim=-1)
            rotation_loss = torch.min(symmetry_losses, dim=-1).values.mean()

        losses.append(rotation_loss)
        losses.append(translation_loss)

        if self.PCD_Loss and source_pcd is not None:
            # source_pcd, predicted_pose, 

            if symmetries is not None:
                target_transform = t_R_sym[torch.arange(symmetries.size(0)), argmin_symmetry] # minimal error target transform

                print(target_transform.shape)
                print(t_T.shape)

            pcd_loss = self.PCD_criterion(source_pcd, predicted_transform, target_transform)
            losses.append(pcd_loss)

        if self.SO_Loss:
            rotation_list = [p_R]
            if extra_SO is not None:
                if type(extra_SO) in [list, tuple]:
                    rotation_list + extra_SO
                else:
                    rotation_list.append(extra_SO)
            ortho_loss = self.SO_criterion(rotation_list)
            losses.append(ortho_loss)

        losses = torch.stack(losses)
        if components:
            return losses
        
        return losses.mean()

class SpecialOrthogonalLoss(_Loss):
    def __init__(self, weight=1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, rotations):

        if type(rotations) not in [list, tuple]:
            rotations = [rotations]
        
        losses = []
        for R in rotations:
            row, col = R.shape[-2:]
            assert row == col
            I = torch.eye(row, device=R.device).expand_as(R)
            ortho_loss = torch.norm(torch.bmm(R, R.transpose(-2, -1)) - I, dim=(-2, -1)).mean() # Over Batch
            losses.append(ortho_loss)
        loss = torch.stack(losses).mean()

        return loss * self.weight
    
class PointCloudMSELoss(_Loss):
    def __init__(self, target_type="pose", weight=1.0) -> None:
        super().__init__()
        self.weight = weight
        assert target_type in ["pose", "point"]
        self.target_type = target_type

    def forward(self, source, pose, target):

        R = pose[:, :, :3]
        T = pose[:, :, 3]

        source_transformed = torch.bmm(source, R.transpose(-2, -1)) + T.unsqueeze(-2)

        if self.target_type == "pose":
            # In this case, target is the ground truth pose
            t_R = target[:, :, :3]
            t_T = target[:, :, 3]
            target = torch.bmm(source, t_R.transpose(-2, -1)) + t_T.unsqueeze(-2)
        # Otherwise, target is a point cloud (WHICH REQUIRES CORRESPONDENCES)

        loss = F.mse_loss(source_transformed, target)

        return loss * self.weight
