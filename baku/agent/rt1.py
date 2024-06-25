# import einops
from einops import rearrange, repeat, reduce, pack, unpack
import numpy as np
from collections import deque

import torch
from torch import nn

from torchvision import transforms as T

import utils
from agent.networks.rgb_modules import ResnetEncoder
from agent.networks.policy_head import RT1Head
from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP


# token learner module
def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        hidden_dim,
        num_feat_per_step=1,
    ):
        super().__init__()

        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        self._policy = GPT(
            GPTConfig(
                block_size=205,  # 110 libero, 205 xarm
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )

        self._action_head = RT1Head(
            hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
        )

        self.apply(utils.weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None):
        B, T, D = obs.shape
        prompt = obs[:, :num_prompt_feats]
        obs = obs[:, num_prompt_feats:]
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
        obs = torch.cat([prompt, obs], dim=1)

        # get action features
        features = self._policy(obs)
        features = features[:, num_prompt_feats:]
        num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
        features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"action_seq": action},
        )

        if action is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
            )
            return pred_action, loss[0] if isinstance(loss, tuple) else loss


class RT1Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        stddev_clip,
        use_tb,
        augment,
        obs_type,
        pixel_keys,
        proprio_key,
        feature_key,
        use_proprio,
        norm,
        history,
        history_len,
        eval_history_len,
        prompt,
        use_language,
        film,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.augment = augment
        self.obs_type = obs_type
        self.use_proprio = use_proprio if obs_type == "pixels" else False
        self.norm = norm
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.use_language = use_language
        self.language_proj_type = "mlp"  # mlp or identity
        self.prompt = prompt
        self.film = film

        # language
        self.language_fusion = "none" if not self.use_language else "film"
        self.language_dim = 384
        self.lang_repr_dim = 512

        # actor parameters
        self._act_dim = action_shape[0]
        # keys
        if obs_type == "pixels":
            self.pixel_keys = pixel_keys
            self.proprio_key = proprio_key
        else:
            self.feature_key = feature_key

        # number of inputs per time step
        if obs_type == "features":
            num_feat_per_step = 1
        elif obs_type == "pixels":
            num_feat_per_step = len(self.pixel_keys) * 8
            if use_proprio:
                num_feat_per_step += 1

        # observation params
        if obs_type == "pixels":
            if use_proprio:
                proprio_shape = obs_shape[self.proprio_key]
            obs_shape = obs_shape[self.pixel_keys[0]]
        else:
            obs_shape = obs_shape[self.feature_key]

        # Track model size
        model_size = 0

        # encoder
        if obs_type == "pixels":
            self.encoder = ResnetEncoder(
                obs_shape,
                512,
                language_dim=self.lang_repr_dim,
                language_fusion=self.language_fusion,
            ).to(device)
            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            self.repr_dim = 512

            # token learner
            self.token_learner = TokenLearner(dim=128).to(device)
            self.image_projector = MLP(128, hidden_channels=[512]).to(device)
        else:
            self.encoder = MLP(obs_shape[0], hidden_channels=[512, 512]).to(device)
            model_size += sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            self.repr_dim = 512
        # language encoder
        if self.use_language:
            # projector
            if self.language_proj_type == "mlp":
                self.language_projector = MLP(
                    self.language_dim,
                    hidden_channels=[self.lang_repr_dim, self.lang_repr_dim],
                ).to(device)
            else:
                self.language_projector = nn.Identity()
            self.language_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.language_projector.parameters()
                if p.requires_grad
            )

        # projector for proprioceptive features
        if use_proprio:
            self.proprio_projector = MLP(
                proprio_shape[0], hidden_channels=[self.repr_dim, self.repr_dim]
            ).to(device)
            self.proprio_projector.apply(utils.weight_init)
            model_size += sum(
                p.numel()
                for p in self.proprio_projector.parameters()
                if p.requires_grad
            )

        self.actor = Actor(
            self.repr_dim,
            self._act_dim,
            hidden_dim,
            num_feat_per_step,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print(f"Total number of parameters in the model: {model_size}")

        # optimizers
        # encoder
        params = list(self.encoder.parameters())
        if self.obs_type == "pixels":
            params += list(self.image_projector.parameters())
        self.encoder_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # proprio
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # language
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=lr, weight_decay=1e-4
            )
        # actor
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )

        # augmentations
        if obs_type == "pixels" and self.norm:
            MEAN = torch.tensor([0.485, 0.456, 0.406])
            STD = torch.tensor([0.229, 0.224, 0.225])
            self.customAug = T.Compose([T.Normalize(mean=MEAN, std=STD)])

        # data augmentation
        if obs_type == "pixels" and self.augment:
            self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "rt1"

    def train(self, training=True):
        self.training = training
        if training:
            self.encoder.train(training)
            if self.use_language:
                self.language_projector.train(training)
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_projector.train(training)
            self.actor.train(training)
        else:
            self.encoder.eval()
            if self.use_language:
                self.language_projector.eval()
            if self.obs_type == "pixels" and self.use_proprio:
                self.proprio_projector.eval()
            self.actor.eval()

    def buffer_reset(self):
        if self.obs_type == "pixels":
            self.observation_buffer = {}
            for key in self.pixel_keys:
                self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
            if self.use_proprio:
                self.proprio_buffer = deque(maxlen=self.eval_history_len)
        else:
            self.observation_buffer = deque(maxlen=self.eval_history_len)

    def clear_buffers(self):
        del self.observation_buffer
        if self.obs_type == "pixels" and self.use_proprio:
            del self.proprio_buffer

    def discretize(self, actions, preprocess):
        print("Discretizing actions ...")

        # organize actions into shape (N, A)
        reshaped_actions = []
        for action in actions:
            action = preprocess["actions"](action)
            reshaped_actions.extend(action)
        reshaped_actions = np.array(reshaped_actions)

        self.actor._action_head.discretize(reshaped_actions, self.device)

        print("Discretization complete.")

    def reinit_optimizers(self):
        params = list(self.encoder.parameters())
        self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        params = list(self.actor.parameters())
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.lr, weight_decay=1e-4
        )

    def act(self, obs, prompt, norm_stats, step, global_step, eval_mode=False):
        if norm_stats is not None:
            pre_process = lambda s_qpos: (
                s_qpos - norm_stats[self.proprio_key]["min"]
            ) / (
                norm_stats[self.proprio_key]["max"]
                - norm_stats[self.proprio_key]["min"]
                + 1e-5
            )
            post_process = (
                lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"]
            )

        # lang projection
        if self.use_language:
            key = self.pixel_keys[0] if self.obs_type == "pixels" else self.feature_key
            repeat_len = (
                min(len(self.observation_buffer[key]) + 1, self.eval_history_len)
                if self.obs_type == "pixels"
                else min(len(self.observation_buffer) + 1, self.eval_history_len)
            )
            lang_features = (
                torch.as_tensor(prompt["task_emb"], device=self.device)
                .float()[None]
                .repeat(repeat_len, 1)
            )
            lang_features = self.language_projector(lang_features)
        else:
            lang_features = None

        if self.obs_type == "pixels":
            # add to buffer
            features = []
            for key in self.pixel_keys:
                self.observation_buffer[key].append(
                    self.test_aug(obs[key].transpose(1, 2, 0)).numpy()
                )
                pixels = torch.as_tensor(
                    np.array(self.observation_buffer[key]), device=self.device
                ).float()
                pixels = self.customAug(pixels / 255.0) if self.norm else pixels
                # encoder
                lang = lang_features if self.film else None
                pixels = self.encoder(pixels, lang=lang, return_intermediate=True)
                pixels = self.token_learner(pixels)
                pixels = rearrange(pixels, "b d k -> b k d")
                pixels = self.image_projector(pixels)
                features.append(pixels)
            if self.use_proprio:
                obs[self.proprio_key] = pre_process(obs[self.proprio_key])
                self.proprio_buffer.append(obs[self.proprio_key])
                proprio = torch.as_tensor(
                    np.array(self.proprio_buffer), device=self.device
                ).float()
                proprio = self.proprio_projector(proprio)
                proprio = proprio[:, None]
                features.append(proprio)
            features = torch.cat(features, dim=-2).view(-1, self.repr_dim)
        else:
            self.observation_buffer.append(obs[self.feature_key])
            features = torch.as_tensor(
                np.array(self.observation_buffer), device=self.device
            ).float()
            features = self.encoder(features)

        # prompt
        prompt_features = []
        if self.use_language:
            prompt_features.append(lang_features[-1:])
        if self.prompt not in [None, "text", "one_hot"]:
            if self.use_language:
                prompt_lang_features = lang_features[-1:]
                reshape_lang = True
            else:
                prompt_lang_features = None

            if self.obs_type == "pixels":
                for key in self.pixel_keys:
                    pixel = torch.as_tensor(
                        prompt[f"prompt_{key}"], device=self.device
                    ).float()
                    shape = pixel.shape
                    # reshape lang features
                    if self.use_language and reshape_lang:
                        prompt_lang_features = prompt_lang_features.repeat(shape[0], 1)
                        reshape_lang = False
                    # augment
                    pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                    # encode
                    pixel = self.encoder(
                        pixel, lang=prompt_lang_features, return_intermediate=True
                    )
                    pixel = self.token_learner(pixel)
                    pixel = rearrange(pixel, "b d k -> b k d")
                    pixel = self.image_projector(pixel)
                    prompt_features.append(pixel)
                if self.use_proprio:
                    proprio = torch.as_tensor(
                        prompt[f"prompt_{self.proprio_key}"], device=self.device
                    ).float()
                    proprio = self.proprio_projector(proprio)
                    proprio = proprio[:, None]
                    prompt_features.append(proprio)
            else:
                prompt_feat = torch.as_tensor(
                    prompt[f"prompt_{self.feature_key}"], device=self.device
                ).float()
                prompt_feat = self.encoder(prompt_feat)
                prompt_features.append(prompt_feat)
        num_prompt_feats = len(prompt_features)
        if num_prompt_feats > 0:
            prompt_features = torch.cat(prompt_features, dim=-1).view(-1, self.repr_dim)
            features = torch.cat([prompt_features, features], dim=0)

        stddev = utils.schedule(self.stddev_schedule, global_step)
        action = self.actor(features.unsqueeze(0), num_prompt_feats, stddev)
        action = action[0]

        if norm_stats is not None:
            return post_process(action.cpu().numpy()[0, -1])
        return action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
        action = data["actions"].float()

        # lang projection
        if self.use_language:
            lang_features = (
                data["task_emb"].float()[:, None].repeat(1, self.history_len, 1)
            )
            lang_features = self.language_projector(lang_features)
            lang_features = rearrange(lang_features, "b t d -> (b t) d")
        else:
            lang_features = None

        # features
        if self.obs_type == "pixels":
            features = []
            for key in self.pixel_keys:
                pixel = data[key].float()
                shape = pixel.shape
                # rearrange
                pixel = rearrange(pixel, "b t c h w -> (b t) c h w")
                # augment
                pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                # encode
                lang = lang_features if self.film else None
                pixel = self.encoder(pixel, lang=lang, return_intermediate=True)

                pixel = self.token_learner(pixel)
                pixel = rearrange(pixel, "b d k -> b k d")
                pixel = self.image_projector(pixel)
                pixel = rearrange(pixel, "(b t) k d -> b t k d", t=shape[1])
                features.append(pixel)
            if self.use_proprio:
                proprio = data[self.proprio_key].float()
                proprio = self.proprio_projector(proprio)
                proprio = proprio[:, :, None]
                features.append(proprio)
            # concatenate
            features = torch.cat(features, dim=-2).view(
                action.shape[0], -1, self.repr_dim
            )  # (B, T * num_feat_per_step, D)
        else:
            features = data[self.feature_key].float()
            shape = features.shape
            features = self.encoder(features)

        # prompt
        prompt_features = []
        if self.use_language:
            lang_features = rearrange(lang_features, "(b t) d -> b t d", t=shape[1])
            prompt_features.append(lang_features[:, -1:])
        if self.prompt not in [None, "text", "one_hot"]:
            if self.use_language:
                prompt_lang_features = lang_features[:, -1:]
                reshape_lang = True
            else:
                prompt_lang_features = None

            if self.obs_type == "pixels":
                for key in self.pixel_keys:
                    pixel = data[f"prompt_{key}"].float()
                    shape = pixel.shape
                    # reshape lang features
                    if self.use_language and reshape_lang:
                        prompt_lang_features = prompt_lang_features.repeat(
                            1, shape[1], 1
                        )
                        prompt_lang_features = rearrange(
                            prompt_lang_features, "b t d -> (b t) d"
                        )
                        reshape_lang = False
                    # rearrange
                    pixel = rearrange(pixel, "b t c h w -> (b t) c h w")
                    # augment
                    pixel = self.customAug(pixel / 255.0) if self.norm else pixel
                    # encode
                    pixel = self.encoder(
                        pixel, lang=prompt_lang_features, return_intermediate=True
                    )
                    pixel = self.token_learner(pixel)
                    pixel = rearrange(pixel, "b d k -> b k d")
                    pixel = self.image_projector(pixel)
                    pixel = rearrange(pixel, "(b t) k d -> b t k d", t=shape[1])
                    prompt_features.append(pixel)
                if self.use_proprio:
                    proprio = data[f"prompt_{self.proprio_key}"].float()
                    proprio = self.proprio_projector(proprio)
                    proprio = proprio[:, :, None]
                    prompt_features.append(proprio)
            else:
                prompt_feat = data[f"prompt_{self.feature_key}"].float()
                prompt_feat = self.encoder(prompt_feat)
                prompt_features.append(prompt_feat)
        num_prompt_feats = len(prompt_features) if len(prompt_features) > 0 else 0
        if num_prompt_feats > 0:
            prompt_features = torch.cat(prompt_features, dim=-2).view(
                action.shape[0], -1, self.repr_dim
            )
            # prepend prompt features
            features = torch.cat([prompt_features, features], dim=1)

        stddev = utils.schedule(self.stddev_schedule, step)
        _, actor_loss = self.actor(features, num_prompt_feats, stddev, action)

        self.encoder_opt.zero_grad(set_to_none=True)
        if self.obs_type == "pixels" and self.use_proprio:
            self.proprio_opt.zero_grad(set_to_none=True)
        if self.use_language:
            self.language_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss["actor_loss"].backward()
        self.encoder_opt.step()
        if self.obs_type == "pixels" and self.use_proprio:
            self.proprio_opt.step()
        if self.use_language:
            self.language_opt.step()
        self.actor_opt.step()

        if self.use_tb:
            for key, value in actor_loss.items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "encoder"]
        if self.obs_type == "pixels":
            model_keys += ["token_learner", "image_projector"]
        opt_keys = ["actor_opt", "encoder_opt"]
        if self.obs_type == "pixels" and self.use_proprio:
            model_keys += ["proprio_projector"]
            opt_keys += ["proprio_opt"]
        if self.use_language:
            model_keys += ["language_projector"]
            opt_keys += ["language_opt"]
        # models
        payload = {
            k: self.__dict__[k].state_dict() for k in model_keys if k != "encoder"
        }
        if "encoder" in model_keys:
            payload["encoder"] = self.encoder.state_dict()
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        # action max and min from rt1
        payload["action_max"] = self.actor._action_head.action_max
        payload["action_min"] = self.actor._action_head.action_min

        others = [
            "use_proprio",
            "use_language",
        ]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=False, load_opt=False):
        # models
        model_keys = ["actor", "encoder"]
        if self.obs_type == "pixels" and self.use_proprio:
            model_keys += ["proprio_projector"]
        if self.use_language:
            model_keys += ["language_projector"]
        for k in model_keys:
            self.__dict__[k].load_state_dict(payload[k])

        # action min and max
        self.actor._action_head.action_max = payload["action_max"]
        self.actor._action_head.action_min = payload["action_min"]

        if eval:
            self.train(False)
            return

        # if not eval
        if not load_opt:
            self.reinit_optimizers()
        else:
            opt_keys = ["actor_opt", "encoder_opt"]
            if self.obs_type == "pixels" and self.use_proprio:
                opt_keys += ["proprio_opt"]
            if self.use_language:
                opt_keys += ["language_opt"]
            if self.use_actions:
                opt_keys += ["action_opt"]
            for k in opt_keys:
                self.__dict__[k] = payload[k]
        self.train(True)
