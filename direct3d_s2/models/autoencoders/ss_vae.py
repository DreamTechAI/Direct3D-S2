# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
from skimage import measure

from ...modules import sparse as sp
from .encoder import SparseSDFEncoder
from .decoder import SparseSDFDecoder
from .distributions import DiagonalGaussianDistribution


class SparseSDFVAE(nn.Module):
    def __init__(self, *,
                 embed_dim: int = 0,
                 resolution: int = 64,
                 model_channels_encoder: int = 512,
                 num_blocks_encoder: int = 4,
                 num_heads_encoder: int = 8,
                 num_head_channels_encoder: int = 64,
                 model_channels_decoder: int = 512,
                 num_blocks_decoder: int = 4,
                 num_heads_decoder: int = 8,
                 num_head_channels_decoder: int = 64,
                 out_channels: int = 1,
                 use_fp16: bool = False,
                 use_checkpoint: bool = False,
                 chunk_size: int = 1,
                 latents_scale: float = 1.0,
                 latents_shift: float = 0.0):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.resolution = resolution
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift

        self.encoder = SparseSDFEncoder(
            resolution=resolution,
            in_channels=model_channels_encoder,
            model_channels=model_channels_encoder,
            latent_channels=embed_dim,
            num_blocks=num_blocks_encoder,
            num_heads=num_heads_encoder,
            num_head_channels=num_head_channels_encoder,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
        )

        self.decoder = SparseSDFDecoder(
            resolution=resolution,
            model_channels=model_channels_decoder,
            latent_channels=embed_dim,
            num_blocks=num_blocks_decoder,
            num_heads=num_heads_decoder,
            num_head_channels=num_head_channels_decoder,
            out_channels=out_channels,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            chunk_size=chunk_size,
        )
        self.embed_dim = embed_dim

    def forward(self, batch):

        z, posterior = self.encode(batch)

        reconst_x = self.decoder(z)
        outputs = {'reconst_x': reconst_x, 'posterior': posterior}
        return outputs

    def encode(self, batch, sample_posterior: bool = True):

        feat, xyz, batch_idx = batch['sparse_sdf'], batch['sparse_index'], batch['batch_idx']
        if feat.ndim == 1:
            feat = feat.unsqueeze(-1)
        coords = torch.cat([batch_idx.unsqueeze(-1), xyz], dim=-1).int()
       
        x = sp.SparseTensor(feat, coords)
        h = self.encoder(x, batch.get('factor', None))
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = h.replace(z)

        return z, posterior     

    def decode_mesh(self,
                    latents,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.2,
                    return_feat: bool = False,
                    factor: float = 1.0):
        voxel_resolution = int(voxel_resolution / factor)
        reconst_x = self.decoder(latents, factor=factor, return_feat=return_feat)
        if return_feat:
            return reconst_x
        outputs = self.sparse2mesh(reconst_x, voxel_resolution=voxel_resolution, mc_threshold=mc_threshold)
        
        return outputs

    def sparse2mesh(self,
                    reconst_x: torch.FloatTensor,
                    voxel_resolution: int = 512,
                    mc_threshold: float = 0.0):

        sparse_sdf, sparse_index = reconst_x.feats.float(), reconst_x.coords
        batch_size = int(sparse_index[..., 0].max().cpu().numpy() + 1)

        meshes = []
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1).cpu(),  sparse_index[idx][..., 1:].detach().cpu()
            sdf = torch.ones((voxel_resolution, voxel_resolution, voxel_resolution))
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i

            # ---------------------------------------------------------------- #
            #   Fix default SDF value for _interior_ voxels via flood-fill.    #
            #   The default for interior voxels should be -1 (not +1).         #
            # ---------------------------------------------------------------- #

            active_mask = torch.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=torch.bool)
            active_mask[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = 1

            sdf = sdf.numpy()
            active_mask = active_mask.numpy()

            # Flood‑fill from object boundary to find interior.
            # Use active voxels to define the object boundary.
            interior_mask = ndi.binary_fill_holes(active_mask)

            # Set inactive voxels to -1 if they are interior
            sdf[interior_mask & (~active_mask)] = -1.0

            vertices, faces, _, _ = measure.marching_cubes(
                sdf,
                mc_threshold,
                method="lewiner",
            )
            vertices = vertices / voxel_resolution * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))

        return meshes
