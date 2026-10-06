import torch
from torch.optim import Optimizer

import math
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class DoG(Optimizer):
    """
     DoG (Distance over Gradients) is a parameter-free adaptive optimizer, proposed in
      `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
    IMPORTANT: for best performance, DoG must be combined with iterate averaging.
    """

    __version__ = "1.0.3"

    def __init__(
        self,
        params,
        reps_rel: float = 1e-6,
        lr: float = 1.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
    ):
        r"""Distance over Gradients - an adaptive stochastic optimizer.

        DoG updates parameters x_t with stochastic gradients g_t according to:
        .. math::
            \begin{aligned}
                eta_t & = \frac{ max_{i \le t}{\|x_i - x_0\|} }{ \sqrt{\sum_{i \le t }{\|g_i\|^2 + eps}} }, \\
                x_{t+1} & = x_{t} - eta_t * g_t,
            \end{aligned}
            
        IMPORTANT: Since we do not employ a step-size decay scheme, ITERATE AVERAGING IS CRUCIAL to obtain 
        the best performance. This package provides an implementation of the polynomial decay averaging 
        that is effective and does not require tuning.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            reps_rel (float): value to use to compute the  initial distance (r_epsilon in the paper).
                                        Namely, the first step size is given by:
                                        (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the initial
                                        weights of  the model (or the parameter group), and g_0 is the gradient of the
                                        first step.
                                        As discussed in the paper, this value should be small enough to ensure that the
                                        first update step will be small enough to not cause the model to diverge.

                                        Suggested value is 1e-6, unless the model uses batch-normalization,
                                        in which case the suggested value is 1e-4. (default: 1e-6)

            lr (float, optional): learning rate (referred to as c in the paper). The default value is 1.0 and changing
                                        it is not recommended.
            weight_decay (float, optional): weight decay (L2 penalty). weight_decay * x_t is added directly
                                            to the gradient (default: 0)
            eps (float, optional): epsilon used for numerical stability - added to the sum of gradients (default: 1e-8)
            init_eta (floar, optional):  if specified, this value will be used the the initial eta (i.e.
                                        first step size), and will override the value of reps_rel (default: None)

        Example:
            >>> optimizer = DoG(model.parameters(), reps_rel=1e-6)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()

        __ https://arxiv.org/pdf/2302.12022.pdf
        """

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate ({lr}). Suggested value is 1.")
        if lr != 1.0:
            logger.warning(
                "We do not recommend changing the lr parameter from its default value of 1"
            )
        if init_eta is not None:
            if init_eta <= 0:
                raise ValueError(f"Invalid value for init_eta ({init_eta})")
            logger.info(
                f"Ignoring reps_rel since will be explicitly set init_eta to be {init_eta} (first step size)"
            )
            reps_rel = 0
        else:
            if reps_rel <= 0.0:
                raise ValueError(
                    f"Invalid reps_rel value ({reps_rel}). Suggested value is 1e-6 "
                    "(unless the model uses batch-normalization, in which case suggested value is 1e-4)"
                )

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self._first_step = True

        defaults = dict(
            reps_rel=reps_rel,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            init_eta=init_eta,
        )
        super(DoG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DoG, self).__setstate__(state)

    def state_dict(self) -> dict:
        state_dict = super(DoG, self).state_dict()
        logger.info("retrieving DoG state dict")
        state_dict["state"]["_first_step"] = self._first_step
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super(DoG, self).load_state_dict(state_dict)
        self._first_step = state_dict["state"]["_first_step"]
        logger.info("loaded DoG state dict")
        cuda = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            cuda_buffers = {"init_buffer"}
            for tgroup in group.keys():
                # this can cast all the tensors to the device. However, as it turns out,
                # we need ONLY the init_buffer to be on the params' device
                if tgroup != "params":
                    device = cuda if tgroup in cuda_buffers else "cpu"
                    if (
                        isinstance(group[tgroup], list)
                        and len(group[tgroup]) > 0
                        and isinstance(group[tgroup][0], torch.Tensor)
                    ):
                        group[tgroup] = [i.to(device) for i in group[tgroup]]
                    elif isinstance(group[tgroup], torch.Tensor):
                        group[tgroup] = group[tgroup].to(device)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._first_step

        for group in self.param_groups:
            weight_decay = group["weight_decay"]

            if first_step:
                init = group["init_buffer"] = [
                    torch.clone(p).detach() for p in group["params"]
                ]
            else:
                init = group["init_buffer"]

            if weight_decay > 0:
                for p in group["params"]:
                    p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            self._override_init_eta_if_needed(group)

            for p, eta in zip(group["params"], group["eta"]):
                if p.grad is None:
                    continue
                else:
                    p.add_(p.grad, alpha=-eta)

        self._first_step = False

        return loss

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            group["rbar"] = group["reps_rel"] * (
                1 + torch.stack([p.norm() for p in group["params"]]).norm()
            )
            group["G"] = (
                torch.stack(
                    [
                        (p.grad.detach() ** 2).sum()
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ).sum()
                + group["eps"]
            )

        else:
            curr_d = torch.stack(
                [torch.norm(p.detach() - pi) for p, pi in zip(group["params"], init)]
            ).norm()
            group["rbar"] = torch.maximum(group["rbar"], curr_d)
            group["G"] += torch.stack(
                [
                    (torch.nan_to_num(p.grad.detach()) ** 2).sum()
                    for p in group["params"]
                ]
            ).sum()

        assert (
            group["G"] > 0
        ), f'DoG cannot work when G is not strictly positive. got: {group["G"]}'
        group["eta"] = [group["lr"] * group["rbar"] / torch.sqrt(group["G"])] * len(
            group["params"]
        )

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group["init_eta"] is not None:
            init_eta = group["init_eta"]
            logger.info(f"Explicitly setting init_eta value to {init_eta}")
            group["eta"] = [eta * 0 + init_eta for eta in group["eta"]]


class UPGD(Optimizer):
    """Utility-based Perturbed Gradient Descent by Elsayed & Mahmood. Implementation taken from https://github.com/mohmdelsayed/upgd/blob/main/core/optim/weight/upgd/first_order.py#L74."""

    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
        )
        super(UPGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_(
                    (state["avg_utility"] / bias_correction) / global_max_util
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-group["lr"],
                )
        return loss


class COCOB(Optimizer):
    def __init__(self, params, lr: float = 100.0, eps: float = 1e-8):
        self.alpha = lr
        self.eps = eps
        defaults = dict(alpha=lr, eps=eps)
        super(COCOB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # optimize for each parameter group
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = -param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state = self.initial_state(state, param)

                initial_weight = state["initial_weight"]
                reward = state["reward"]
                bet = state["bet"]
                grads_sum = state["grads_sum"]
                absolute_grads_sum = state["absolute_grads_sum"]
                max_observed_scale = state["max_observed_scale"]

                # update parameters
                abs_grad = torch.abs(grad)
                max_observed_scale = torch.max(max_observed_scale, abs_grad)
                absolute_grads_sum += abs_grad
                grads_sum += grad

                win_amount = bet * grad
                reward = torch.max(reward + win_amount, torch.zeros_like(reward))
                bet_fraction = grads_sum / (
                    max_observed_scale
                    * (
                        torch.max(
                            absolute_grads_sum + max_observed_scale,
                            self.alpha * max_observed_scale,
                        )
                    )
                )
                bet = bet_fraction * (max_observed_scale + reward)

                # set new state
                param.data = initial_weight + bet
                state["grads_sum"] = grads_sum
                state["absolute_grads_sum"] = absolute_grads_sum
                state["max_observed_scale"] = max_observed_scale
                state["reward"] = reward
                state["bet"] = bet

        return loss

    def initial_state(self, state, param):
        assert len(state) == 0

        state["initial_weight"] = param.data
        state["reward"] = param.new_zeros(param.shape)
        state["bet"] = param.new_zeros(param.shape)
        state["grads_sum"] = param.new_zeros(param.shape)
        state["absolute_grads_sum"] = param.new_zeros(param.shape)
        state["max_observed_scale"] = self.eps * param.new_ones(param.shape)

        return state


class AdamHD(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        hypergrad_lr=1e-8,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hypergrad_lr=hypergrad_lr,
        )
        super(AdamHD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                if state["step"] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state["step"] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state["step"] - 1)
                    # Hypergradient for Adam:
                    h = (
                        torch.dot(
                            grad.view(-1),
                            torch.div(
                                exp_avg, exp_avg_sq.sqrt().add_(group["eps"])
                            ).view(-1),
                        )
                        * math.sqrt(prev_bias_correction2)
                        / prev_bias_correction1
                    )
                    # Hypergradient descent of the learning rate:
                    group["lr"] += group["hypergrad_lr"] * h

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class WNGrad(Optimizer):
    def __init__(self, params, lr=0.05) -> None:
        self.scale = lr
        self.scale_sq = lr**2
        self._step_count = 1
        defaults = dict(scale=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._step_count += 1

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state = self.initial_state(state, param)

                state["b"] = (
                    state["b"]
                    + self.scale_sq
                    * torch.dot(grad.view(-1), grad.view(-1))
                    / state["b"]
                )
                lr = self.scale / state["b"]
                param.data.sub_(lr * grad)
            group["lr"] = lr
        return loss

    def initial_state(self, state, param):
        assert len(state) == 0
        state["b"] = param.new_ones(1)
        return state
