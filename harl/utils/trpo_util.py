"""TRPO utility functions."""
import torch


def flat_grad(grads):
    """Flatten the gradients."""
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).view(-1))
    if not grad_flatten:
        return torch.empty(0)
    return torch.cat(grad_flatten)


def flat_hessian(hessians):
    """Flatten the hessians."""
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        clean_hessian = torch.nan_to_num(hessian, nan=0.0, posinf=0.0, neginf=0.0)
        hessians_flatten.append(clean_hessian.contiguous().view(-1))
    if not hessians_flatten:
        return torch.empty(0)
    return torch.cat(hessians_flatten).data


def flat_params(model):
    """Flatten the parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """Update the model parameters."""
    if not torch.isfinite(new_params).all():
        raise ValueError("TRPO recibio parametros no finitos.")
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_approx(p, q):
    """KL divergence between two distributions."""
    r = torch.exp(q - p)
    kl = r - 1 - q + p
    return kl


def _kl_normal_normal(p, q):
    """KL divergence between two normal distributions.
    adapted from https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
    """
    p_scale = torch.clamp(p.scale.to(torch.float64), min=1e-6, max=10.0)
    q_scale = torch.clamp(q.scale.to(torch.float64), min=1e-6, max=10.0)
    var_ratio = (p_scale / q_scale).pow(2)
    t1 = (
        (p.loc.to(torch.float64) - q.loc.to(torch.float64)) / q_scale
    ).pow(2)
    kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    return torch.nan_to_num(kl.to(torch.float32), nan=0.0, posinf=1e6, neginf=0.0)


def kl_divergence(
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    new_actor,
    old_actor,
):
    """KL divergence between two distributions."""
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
    if new_dist.__class__.__name__ == "FixedCategorical":  # discrete action
        new_logits = new_dist.logits
        old_logits = old_dist.logits
        kl = kl_approx(old_logits, new_logits)
    else:  # continuous action
        kl = _kl_normal_normal(old_dist, new_dist)

    if len(kl.shape) > 1:
        kl = kl.sum(1, keepdim=True)
    return torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0)


# pylint: disable-next=invalid-name
def conjugate_gradient(
    actor,
    obs,
    rnn_states,
    action,
    masks,
    available_actions,
    active_masks,
    b,
    nsteps,
    device,
    residual_tol=1e-10,
):
    """Conjugate gradient algorithm.
    # refer to https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    """
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.zeros(b.size()).to(device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    if (not torch.isfinite(rdotr)) or rdotr <= residual_tol:
        return x
    for _ in range(nsteps):
        _Avp = fisher_vector_product(
            actor, obs, rnn_states, action, masks, available_actions, active_masks, p
        )
        denom = torch.dot(p, _Avp)
        if (not torch.isfinite(denom)) or torch.abs(denom) <= 1e-12:
            break
        alpha = rdotr / denom
        if not torch.isfinite(alpha):
            break
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        if not torch.isfinite(new_rdotr):
            break
        betta = new_rdotr / rdotr
        if not torch.isfinite(betta):
            break
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fisher_vector_product(
    actor, obs, rnn_states, action, masks, available_actions, active_masks, p
):
    """Fisher vector product."""
    with torch.backends.cudnn.flags(enabled=False):
        p = torch.nan_to_num(p.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        kl = kl_divergence(
            obs,
            rnn_states,
            action,
            masks,
            available_actions,
            active_masks,
            new_actor=actor,
            old_actor=actor,
        )
        kl = kl.mean()
        kl_grad = torch.autograd.grad(
            kl, actor.parameters(), create_graph=True, allow_unused=True
        )
        kl_grad = flat_grad(kl_grad)  # check kl_grad == 0
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(
            kl_grad_p, actor.parameters(), allow_unused=True
        )
        kl_hessian_p = flat_hessian(kl_hessian_p)
        if kl_hessian_p.numel() != p.numel():
            return 0.1 * p
        return torch.nan_to_num(kl_hessian_p + 0.1 * p, nan=0.0, posinf=0.0, neginf=0.0)
