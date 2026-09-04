import torch

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    """
    Returns: dictionary containing new_param and full_grad tensors
    """
    p = param.detach().clone().requires_grad_(True)
    total_samples = sum(len(inputs) for inputs in microbatch_inputs)
    
    for x_m, y_m in zip(microbatch_inputs, microbatch_targets):
        if not isinstance(x_m, torch.Tensor):
            x_m = torch.tensor(x_m, dtype=param.dtype, device=param.device)
        else:
            x_m = x_m.to(dtype=param.dtype, device=param.device)
            
        if not isinstance(y_m, torch.Tensor):
            y_m = torch.tensor(y_m, dtype=param.dtype, device=param.device)
        else:
            y_m = y_m.to(dtype=param.dtype, device=param.device)
            
        N_m = x_m.shape[0]
        if N_m == 0:
            continue
        preds = x_m @ p
        loss_m = torch.nn.functional.mse_loss(preds, y_m)

        scaled_loss_m = (N_m / total_samples) * loss_m
        scaled_loss_m.backward()
        
    full_grad = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(param)
    
    with torch.no_grad():
        new_param = param - lr * full_grad
        
    return {
        "new_param": new_param,
        "full_grad": full_grad
    }