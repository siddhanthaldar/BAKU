import tqdm
import torch


class KMeansDiscretizer:
    """
    Simplified and modified version of KMeans algorithm from sklearn.

    Code borrowed from https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(
        self,
        num_bins: int = 100,
        kmeans_iters: int = 50,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.kmeans_iters = kmeans_iters

    def fit(self, input_actions: torch.Tensor) -> None:
        self.bin_centers = KMeansDiscretizer._kmeans(
            input_actions, nbin=self.n_bins, niter=self.kmeans_iters
        )

    @classmethod
    def _kmeans(cls, x: torch.Tensor, nbin: int = 512, niter: int = 50):
        """
        Simple k-means bining algorithm adapted from Karpathy's minGPT libary
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:nbin]]  # init bins at random

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means bining")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(nbin)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead bins"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead bins
        return c
