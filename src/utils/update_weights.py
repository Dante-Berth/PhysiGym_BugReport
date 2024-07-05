import torch


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: int) -> None:
    """
    Update the target network parameters with soft updates.

    This method updates the parameters of the target network gradually towards the parameters of the source network,
    following the formula:

    target_params = target_params * (1 - tau) + source_params * tau

    This process is commonly known as a "soft" or "smooth" update. It is akin to a form of moving average update, where
    the parameters of the target network are blended with those of the source network using a coefficient (tau).
    This helps stabilize training by smoothing out the changes over time.

    Args:
        source (torch.nn.Module): The source network from which parameters are copied.
        target (torch.nn.Module): The target network whose parameters will be updated.
        tau (int): The soft update coefficient, controlling the rate of update. Values closer to 0 result in slower updates,
                while values closer to 1 result in faster updates. A typical value is in the range (0, 1).
    """
    # Check if the number of parameters in the target and source networks are the same
    if sum(p.numel() for p in target.parameters()) != sum(
        p.numel() for p in source.parameters()
    ):
        raise ValueError(
            "The number of parameters in the target and source networks must be the same for soft update."
        )
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
