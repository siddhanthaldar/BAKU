import einops
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.networks.utils.diffusion_policy import DiffusionPolicy
from agent.networks.utils.vqbet.pretrain_vqvae import init_vqvae, pretrain_vqvae
from agent.networks.mlp import MLP

######################################### Deterministic Head #########################################


class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
        }


######################################### Gaussian Mixture Model Head #########################################


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x, **kwargs):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x, stddev=None, **kwargs):
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean", **kwargs):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
        return {
            "actor_loss": loss,
        }


######################################### BeT Head #########################################


class FocalLoss(nn.Module):
    """
    From https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(N, T, 1)).view(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BeTHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        nbins=64,
        cluster_centers=None,
        # loss_kwargs
        offset_loss_weight=100.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.cluster_centers = cluster_centers
        self.offset_loss_weight = offset_loss_weight

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        # Bin head
        self.bin_head = nn.Sequential(nn.Linear(hidden_size, nbins))

        # Offset head
        self.repeat_action = hidden_size // self.output_size
        self.offset_head = nn.Sequential(
            nn.Linear(
                hidden_size + self.repeat_action * self.output_size, self.output_size
            )
        )

        # loss
        self.criterion = FocalLoss(gamma=2.0)

    def find_closest_cluster(self, actions, cluster_centers) -> torch.Tensor:
        N, T, _ = actions.shape
        actions = einops.rearrange(actions, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # N K A -> N K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N )
        closest_cluster_center = closest_cluster_center.view(N, T)
        return closest_cluster_center

    def forward(self, x, stddev=None, cluster_centers=None, **kwargs):
        feat = self.share(x)

        # Bin head
        bin_logits = self.bin_head(feat)

        # get base action
        N, T, choices = bin_logits.shape
        if N > 1:
            # For training, always take the best action
            sampled_center = torch.argmax(bin_logits, dim=-1, keepdim=True)
        else:
            # Sample center based on login probability
            sampled_center = D.Categorical(logits=bin_logits).sample()
        base_action = cluster_centers[sampled_center.flatten()]
        repeated_base_action = base_action.view(N, T, -1).repeat(
            1, 1, self.repeat_action
        )

        # Offset head
        h = torch.cat([feat, repeated_base_action], dim=-1)
        offset = self.offset_head(h)

        return (bin_logits, offset, base_action)

    def loss_fn(self, pred, target, reduction="mean", cluster_centers=None):
        bin_logits, offset, _ = pred

        # Get expert logits and offsets

        true_bins = self.find_closest_cluster(target, cluster_centers)
        true_offsets = target - cluster_centers[true_bins]

        # loss
        discrete_loss = self.criterion(bin_logits, true_bins)
        offset_loss = F.mse_loss(offset, true_offsets)
        actor_loss = discrete_loss + self.offset_loss_weight * offset_loss

        return {
            "actor_loss": actor_loss,
            "discrete_loss": discrete_loss,
            "offset_loss": offset_loss,
        }


######################################### VQ-BeT Head #########################################


class VQBeTHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        device="cuda",
        # loss_kwargs
        offset_loss_weight=100.0,
        secondary_code_multiplier=0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_multiplier = secondary_code_multiplier
        self.sequentially_select = True

        # init vqvae
        config = {
            "action_dim": output_size,
            "device": device,
        }
        self._vqvae_model = init_vqvae(config)
        self._G = self._vqvae_model.vqvae_groups  # G(number of groups)
        self._C = self._vqvae_model.vqvae_n_embed  # C(number of code integers)
        self._D = self._vqvae_model.embedding_dim  # D(embedding dims)

        if self.sequentially_select:
            print("use sequantial prediction for vq dictionary!")
            self._map_to_cbet_preds_bin1 = MLP(
                in_channels=self.input_size,
                hidden_channels=[self.hidden_size, self.hidden_size, self._C],
            ).to(config["device"])
            self._map_to_cbet_preds_bin2 = MLP(
                in_channels=self.input_size + self._C,
                hidden_channels=[self.hidden_size, self._C],
            ).to(config["device"])
        else:
            self._map_to_cbet_preds_bin = MLP(
                in_channels=self.input_size,
                hidden_channels=[self.hidden_size, self.hidden_size, self._G * self._C],
            ).to(config["device"])
        self._map_to_cbet_preds_offset = MLP(
            in_channels=self.input_size,
            hidden_channels=[
                self.hidden_size,
                self.hidden_size,
                self._G * self._C * self.output_size,
            ],
        ).to(config["device"])

        # loss
        self._criterion = FocalLoss(gamma=2.0)

    def discretize(self, config, actions):
        self._vqvae_model = pretrain_vqvae(self._vqvae_model, config, actions)
        self._vqvae_model.eval()
        for param in self._vqvae_model.vq_layer.parameters():
            param.requires_grad = False

    def forward(self, x, stddev=None, **kwargs):
        N, T, _ = x.shape
        x = einops.rearrange(x, "N T WA -> (N T) WA")

        if self.sequentially_select:
            cbet_logits1 = self._map_to_cbet_preds_bin1(x)
            cbet_offsets = self._map_to_cbet_preds_offset(x)
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs1 = torch.softmax(cbet_logits1, dim=-1)
            NT, choices = cbet_probs1.shape
            G = self._G
            sampled_centers1 = einops.rearrange(
                torch.multinomial(cbet_probs1.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (x, F.one_hot(sampled_centers1, num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_probs2 = torch.softmax(cbet_logits2, dim=-1)
            sampled_centers2 = einops.rearrange(
                torch.multinomial(cbet_probs2.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack(
                (sampled_centers1, sampled_centers2), axis=1
            )  # NT, G
        else:
            cbet_logits = self._map_to_cbet_preds_bin(x)
            cbet_offsets = self._map_to_cbet_preds_offset(x)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
            )
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs = torch.softmax(cbet_logits, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        indices = (
            torch.arange(NT).unsqueeze(1).cuda(),
            torch.arange(self._G).unsqueeze(0).cuda(),
            sampled_centers,
        )
        # Use advanced indexing to sample the values
        sampled_offsets = cbet_offsets[indices]  # NT, G, W, A(?) or NT, G, A

        sampled_offsets = sampled_offsets.sum(dim=1)
        centers = self._vqvae_model.draw_code_forward(sampled_centers).view(
            NT, -1, self._D
        )
        return_decoder_input = einops.rearrange(
            centers.clone().detach(), "NT G D -> NT (G D)"
        )
        decoded_action = (
            self._vqvae_model.get_action_from_latent(return_decoder_input)
            .clone()
            .detach()
        )  # NT, A
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self._vqvae_model.input_dim_h
        )
        predicted_action = decoded_action + sampled_offsets
        predicted_action = einops.rearrange(
            predicted_action,
            "(N T) W A -> N T (W A)",
            N=N,
            T=T,
            W=self._vqvae_model.input_dim_h,
        )

        return {
            "input": x,
            "cbet_logits1": cbet_logits1 if "cbet_logits1" in locals() else None,
            "cbet_logits2": cbet_logits2 if "cbet_logits2" in locals() else None,
            "cbet_logits": cbet_logits if "cbet_logits" in locals() else None,
            "predicted_action": predicted_action,
            "decoded_action": decoded_action,
            "sampled_centers": sampled_centers,
            "G": G,
            "NT": NT,
            "N": N,
            "T": T,
        }

    def loss_fn(self, pred, target, **kwargs):
        # Rename the inputs for clarity.
        action_seq = target
        gpt_output = pred["input"]
        predicted_action = pred["predicted_action"]
        decoded_action = pred["decoded_action"]
        sampled_centers = pred["sampled_centers"]
        G, NT, N, T = pred["G"], pred["NT"], pred["N"], pred["T"]
        if self.sequentially_select:
            cbet_logits1 = pred["cbet_logits1"]
            cbet_logits2 = pred["cbet_logits2"]
        else:
            cbet_logits = pred["cbet_logits"]

        predicted_action = einops.rearrange(
            predicted_action, "N T (W A) -> (N T) W A", W=self._vqvae_model.input_dim_h
        )

        n, total_w, act_dim = action_seq.shape
        act_w = self._vqvae_model.input_dim_h
        obs_w = total_w + 1 - act_w
        output_shape = (n, obs_w, act_w, act_dim)
        output = torch.empty(output_shape).to(action_seq.device)
        for i in range(obs_w):
            output[:, i, :, :] = action_seq[:, i : i + act_w, :]
        action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
        # Figure out the loss for the actions.
        # First, we need to find the closest cluster center for each action.
        state_vq, action_bins = self._vqvae_model.get_code(
            action_seq
        )  # action_bins: NT, G

        # Now we can compute the loss.
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        offset_loss = torch.nn.L1Loss()(action_seq, predicted_action)

        action_diff = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[:, -1, 0, :],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, 0, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_tot = F.mse_loss(
            einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[:, -1, :, :],
            einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                :, -1, :, :
            ],
        )  # batch, time, windowsize (t ... t+N), action dim -> [:, -1, 0, :] is for rollout
        action_diff_mean_res1 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(decoded_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
            )
        ).mean()
        action_diff_mean_res2 = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
            )
        ).mean()
        action_diff_max = (
            abs(
                einops.rearrange(action_seq, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
                - einops.rearrange(predicted_action, "(N T) W A -> N T W A", T=obs_w)[
                    :, -1, 0, :
                ]
            )
        ).max()

        if self.sequentially_select:
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits1[:, :],
                action_bins[:, 0],
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits2[:, :],
                action_bins[:, 1],
            )
        else:
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 0, :],
                action_bins[:, 0],
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 1, :],
                action_bins[:, 1],
            )
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self.secondary_code_multiplier

        equal_total_code_rate = (
            torch.sum(
                (torch.sum((action_bins == sampled_centers).int(), axis=1) == G).int()
            )
            / NT
        )
        equal_single_code_rate = torch.sum(
            (action_bins[:, 0] == sampled_centers[:, 0]).int()
        ) / (NT)
        equal_single_code_rate2 = torch.sum(
            (action_bins[:, 1] == sampled_centers[:, 1]).int()
        ) / (NT)

        loss = cbet_loss + self.offset_loss_weight * offset_loss
        loss_dict = {
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item(),
            "equal_total_code_rate": equal_total_code_rate,
            "equal_single_code_rate": equal_single_code_rate,
            "equal_single_code_rate2": equal_single_code_rate2,
            "action_diff": action_diff.detach().cpu().item(),
            "action_diff_tot": action_diff_tot.detach().cpu().item(),
            "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
            "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
            "action_diff_max": action_diff_max.detach().cpu().item(),
        }
        return {"actor_loss": loss}, loss_dict


######################################### Diffusion Head #########################################


class DiffusionHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        obs_horizon,  # history len
        pred_horizon,
        hidden_size=1024,
        num_layers=2,
        device="cpu",
        loss_coef=100.0,
    ):
        super().__init__()

        self.net = DiffusionPolicy(
            obs_dim=input_size,
            act_dim=output_size,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            policy_type="transformer",
            device=device,
        )

        self.loss_coef = loss_coef

    def forward(self, x, stddev=None, **kwargs):
        return self.net(x, kwargs.get("action_seq", None))

    def loss_fn(self, out, target, reduction="mean", **kwargs):
        noise_pred = out["noise_pred"]
        noise = out["noise"]

        return {
            "actor_loss": F.mse_loss(noise_pred, noise, reduction=reduction)
            * self.loss_coef,
        }


######################################### RT-1 Head #########################################


class RT1Head(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        nbins=256,
    ):
        super().__init__()
        self.output_size = output_size
        self.nbins = nbins

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        # Bin head
        self.bin_head = nn.Sequential(nn.Linear(hidden_size, output_size * nbins))

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # initialize action max and min for discretization
        self.action_max, self.action_min = None, None

    def find_closest_cluster(self, actions, cluster_centers) -> torch.Tensor:
        N, T, _ = actions.shape
        actions = einops.rearrange(actions, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # N K A -> N K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N )
        closest_cluster_center = closest_cluster_center.view(N, T)
        return closest_cluster_center

    def forward(self, x, stddev=None, cluster_centers=None, **kwargs):
        feat = self.share(x)

        # Bin head
        bin_logits = self.bin_head(feat)

        # discretize each action dim
        bin_logits = einops.rearrange(bin_logits, "N T (A K) -> N T A K", K=self.nbins)
        # bin_logits = torch.softmax(bin_logits, dim=-1)

        return self.discrete_to_continuous(bin_logits), bin_logits

    def discretize(self, actions, device):
        actions = torch.tensor(actions)
        self.action_max = torch.max(actions, dim=0)[0].to(device)
        self.action_min = torch.min(actions, dim=0)[0].to(device)

    def discrete_to_continuous(self, action_logits):
        action_logits = torch.argmax(action_logits, dim=-1)
        action_logits = action_logits.float()
        action_logits = (action_logits / (self.nbins - 1)) * (
            self.action_max - self.action_min
        ) + self.action_min
        return action_logits

    def continuous_to_discrete(self, actions):
        actions = (actions - self.action_min) / (self.action_max - self.action_min)
        actions = actions * (self.nbins - 1)
        actions = actions.round()
        return actions

    def loss_fn(self, action, gt_actions, reduction="mean", cluster_centers=None):
        _, action_logits = action

        gt_actions = self.continuous_to_discrete(gt_actions)
        # rearrage for cross entropy loss
        gt_actions = einops.rearrange(gt_actions, "N T A -> (N T) A").long()
        action_logits = einops.rearrange(action_logits, "N T A K -> (N T) K A")

        # loss
        loss = self.criterion(action_logits, gt_actions)

        return {
            "actor_loss": loss,
        }
