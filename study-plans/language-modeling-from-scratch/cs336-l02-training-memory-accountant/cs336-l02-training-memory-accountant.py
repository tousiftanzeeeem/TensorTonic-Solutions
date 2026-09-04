import math

def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element,
                        activation_shapes, activation_bytes_per_element,
                        optimizer, optimizer_bytes_per_element):
    """
    Returns: dictionary containing exact parameter, gradient, activation, optimizer, and total bytes
    """
    params = 0
    for shape in param_shapes:
        tmp = 1
        for i in shape:
            tmp = tmp * i
        params = params + tmp
        
    activation_cnt = 0
    for shape in activation_shapes:
        tmp = 1
        for i in shape:
            tmp = tmp * i
        activation_cnt = activation_cnt + tmp
        
    parameter_byte = params * param_bytes_per_element
    gradients = params * grad_bytes_per_element
    activations = activation_cnt * activation_bytes_per_element
    
    # Determine the number of optimizer state tensors per parameter element
    if optimizer == "sgd":
        opt_states_count = 0
    elif optimizer == "adagrad":
        opt_states_count = 1
    elif optimizer == "adam":
        opt_states_count = 2
    else:
        opt_states_count = 0
        
    optimizer_state = params * opt_states_count * optimizer_bytes_per_element
    total = parameter_byte + gradients + activations + optimizer_state
    
    return {
        "parameters": parameter_byte,
        "gradients": gradients,
        "activations": activations,
        "optimizer_state": optimizer_state,
        "total": total
    }