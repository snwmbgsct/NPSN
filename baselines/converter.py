import torch


def get_sgcn_identity(shape):
    identity_spatial = torch.ones((shape[1], shape[2], shape[2]), device='cuda') * torch.eye(shape[2], device='cuda')
    identity_temporal = torch.ones((shape[2], shape[1], shape[1]), device='cuda') * torch.eye(shape[1], device='cuda')
    return [identity_spatial, identity_temporal]  # [obs_len N N], [N obs_len obs_len]

# identity_spatial is initialized with dimensions (shape[1], shape[2], shape[2]) and is filled with the identity matrix with dimensions (shape[2], shape[2]). Similarly, identity_temporal is initialized with dimensions (shape[2], shape[1], shape[1]) and is filled with the identity matrix with dimensions (shape[1], shape[1]).