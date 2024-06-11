import os
import einops

# import hydra
import numpy as np
import torch
import tqdm
from pathlib import Path
from agent.networks.utils.vqbet.vqvae import VqVae


# PyTorch dataset class for loading actions
class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, actions):
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.actions[idx]


def init_vqvae(config):
    # model
    vqvae_model = VqVae(
        obs_dim=41,
        input_dim_h=1,  # since actions are passed in chunked
        input_dim_w=config["action_dim"],
        n_latent_dims=256, 
        vqvae_n_embed=16,
        vqvae_groups=2,
        eval=False,
        device=config["device"],
    )

    return vqvae_model


def pretrain_vqvae(vqvae_model, config, actions):
    # logger
    from logger import Logger

    # create logger
    logger = Logger(Path("."), use_tb=True, mode="ssl")

    # data
    train_data = ActionDataset(actions)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=False
    )

    for epoch in range(config["epochs"]):
        for act in train_loader:
            act = act.to(vqvae_model.device)

            (
                encoder_loss,
                vq_loss_state,
                vq_code,
                vqvae_recon_loss,
                rep_loss,
            ) = vqvae_model.vqvae_update(
                act
            )  # N T D

            metrics = {
                "encoder_loss": encoder_loss,
                "vq_loss_state": vq_loss_state,
                "vqvae_recon_loss": vqvae_recon_loss,
                "loss": rep_loss,
                "n_different_codes": len(torch.unique(vq_code)),
                "n_different_combinations": len(torch.unique(vq_code, dim=0)),
            }
            logger.log_metrics(metrics, epoch, ty="train_vq")

        if epoch % config["save_every"] == 0:
            # log
            with logger.log_and_dump_ctx(epoch, ty="train_vq") as log:
                log("loss", rep_loss)
                log("step", epoch)

            # save weight
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(".", "trained_vqvae.pt"))

    return vqvae_model
