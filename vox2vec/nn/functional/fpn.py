from typing import *

import torch


def select_from_pyramid(
        feature_pyramid: Sequence[torch.Tensor],
        indices: torch.Tensor,
) -> torch.Tensor:
    """Select features from feature pyramid by their indices w.r.t. base feature map.

    Args:
        feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(c_i, h_i, w_i, d_i)``.
        indices (torch.Tensor): tensor of shape ``(n, 3)``

    Returns:
        torch.Tensor: tensor of shape ``(n, \sum_i c_i)``
    """
    # s = torch.cat([x.moveaxis(0, -1)[(indices // 2 ** i).unbind(1)] for i, x in enumerate(feature_pyramid)], dim=1)
    s = torch.cat([x.moveaxis(0, -1)[(indices // 2 ** 2).unbind(1)] for i, x in enumerate(feature_pyramid[1].unsqueeze(0))], dim=1)
    # Combine specific parts of tensors from feature_pyramid[0]
#     slices = []
#     for i, tensor in enumerate(feature_pyramid):
#         modified_tensor = tensor.transpose(0, -1)  # Swap first and last dimensions
#         # selected_slice = modified_tensor[indices[i] // 4]  # Extract a specific part
#         selected_slice = modified_tensor[indices[i] // 2**i]  # Extract a specific part
#         slices.append(selected_slice)

# # Concatenate the selected slices along dimension 1
#     s = torch.cat(slices, dim=1)

    return s


def sum_pyramid_channels(base_channels: int, num_scales: int):
    # return 16
    return sum(base_channels * 2 ** i for i in range(num_scales))
