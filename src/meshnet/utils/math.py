import torch


def stress_tensor_to_voigt(stress_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a stress tensor to Voigt notation.

    Args:
        stress_tensor (torch.Tensor): A tensor of shape (..., 3, 3) representing the stress tensor.

    Returns:
        torch.Tensor: A tensor of shape (..., 6) representing the stress in Voigt notation.
    """
    s_xx = stress_tensor[..., 0, 0]
    s_yy = stress_tensor[..., 1, 1]
    s_zz = stress_tensor[..., 2, 2]
    s_xy = stress_tensor[..., 0, 1]
    s_yz = stress_tensor[..., 1, 2]
    s_zx = stress_tensor[..., 2, 0]

    return torch.stack((s_xx, s_yy, s_zz, s_xy, s_yz, s_zx), dim=-1)


def calculate_von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Calculate the von Mises stress from the stress tensor.

    The stress tensor is expected to be in the form of a 6-component vector for each point,
    representing the components of the stress tensor in Voigt notation: [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx].

    Args:
        stress (torch.Tensor): The stress tensor of shape (..., 6).

    Returns:
        torch.Tensor: The von Mises stress of shape (...,).
    """
    # Extract the stress components
    s_xx, s_yy, s_zz = stress[:, 0], stress[:, 1], stress[:, 2]
    t_xy, t_yz, t_zx = stress[:, 3], stress[:, 4], stress[:, 5]
    return torch.sqrt(
        0.5
        * (
            (s_xx - s_yy) ** 2
            + (s_yy - s_zz) ** 2
            + (s_zz - s_xx) ** 2
            + 6 * (t_xy**2 + t_yz**2 + t_zx**2)
        )
    )
