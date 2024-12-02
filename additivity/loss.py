import torch as t


def disentanglement_loss(ae, x):
    half_batch_dim = x.shape[0] // 2
    x = x[: half_batch_dim * 2].to(ae.decoder.weight.device)
    random_indices = t.randperm(half_batch_dim * 2)
    x1 = x[random_indices[:half_batch_dim]]
    x2 = x[random_indices[half_batch_dim:]]
    x_composed = x1 + x2  # ! should I devide by 2?
    x_composed.requires_grad = True

    _, f1_gate = ae.encode(x1, return_gate=True)
    _, f2_gate = ae.encode(x2, return_gate=True)
    _, f_composed_gate = ae.encode(x_composed, return_gate=True)

    x1_gate = f1_gate @ ae.decoder.weight.detach().T + ae.decoder_bias.detach()
    x2_gate = f2_gate @ ae.decoder.weight.detach().T + ae.decoder_bias.detach()
    x_composed_gate = (
        f_composed_gate @ ae.decoder.weight.detach().T + ae.decoder_bias.detach()
    )

    # Reduce composition reconstruction error: gate(h1 + h2) - gate(h_composed) == 0
    composition_loss = (x_composed_gate - x1_gate - x2_gate).pow(2).mean(dim=-1).sum()

    # Reduce disentanglement error: gate(h1) intersect gate(h2) == {}
    disentanglement_loss = (f1_gate * f2_gate).sum(dim=-1).mean()

    # if logging:
    #     print(f"{composition_loss.item()=} | {disentanglement_loss.item()=}")

    return composition_loss + disentanglement_loss

def additivity_loss(ae, x):
    half_batch_dim = x.shape[0] // 2
    x = x[: half_batch_dim * 2].to(ae.decoder.weight.device)
    random_indices = t.randperm(half_batch_dim * 2)
    x1 = x[random_indices[:half_batch_dim]]
    x2 = x[random_indices[half_batch_dim:]]
    x_composed = x1 + x2  # ! should I devide by 2?
    x_composed.requires_grad = True
    f1 = ae.encode(x1)
    f2 = ae.encode(x2)
    f_composed = ae.encode(x_composed)

    # Reduce additivity error: f(h1) + f(h2) == f(h1 + h2)
    additivity_loss = (f1 + f2 - f_composed).pow(2).mean(dim=-1).sum()
    return additivity_loss