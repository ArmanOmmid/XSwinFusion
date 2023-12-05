
from typing import *
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._network import _Network
from .modules import SwinTransformerBlockV2_Modulated, ViTEncoderBlock_Modulated, \
                        PatchMergingV2_Modulated, PatchExpandingV2_Modulated, Patching_Modulated, UnPatching_Modulated, \
                        SwinResidualCrossAttention_Modulated, ConvolutionTriplet_Modulated, PointwiseConvolution_Modulated, \
                        create_positional_embedding, initialize_weights, \
                        TimestepEmbedder, LabelEmbedder, initalize_diffusion, ConditionedSequential

# TODO : Replace torch.permute() with (batch) dim agnostic Tensor.transpose()

class XNetSwinTransformerDiffusion(_Network):
    """
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.

        global_stages (int): Global Attention ViT Layers between Encoder and Decoder
        input_size (List[int]): Gives input size of the data. If not provided, Global ViT Layers will NOT have positional embeddings
        final_downsample (bool): Do a final downsampling for the encoder towards the mid-network. If there are no Global ViT Layers, this is ignored.
        cross_attention_residual (bool): Use cross attention for Swin residual connections
        weights (str): Path to load weights
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-5),
        global_stages: int = 1, 
        input_size: List[int] = None, # Needed to deduce the positional encodings in the global ViT layers
        final_downsample: bool = True,
        residual_cross_attention: bool = True,
        input_channels=None,
        output_channels=None,
        class_dropout_prob=0.1,
        smooth_conv=True,
        weights=None,
    ):
        super().__init__()

        if input_channels is None or output_channels is None:
            raise ValueError("Please Provide The Input/Output Diffusion Dimensionality")
        
        if input_size is None and global_stages > 0:
            print("WARNING: Not Providing The Argument 'input_size' Means Global ViT Blocks will NOT Have Positional Embedings")

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.window_size = window_size
        self.residual_cross_attention = residual_cross_attention

        self.smooth_conv = smooth_conv
        self.has_global_stages = global_stages > 0
        self.final_downsample = final_downsample and self.has_global_stages
        total_stage_blocks = sum(depths)

        # Diffusion Embeddings

        self.mod_dims = embed_dim
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)

        # Smooth Patch Partitioning
        if self.smooth_conv:
            self.smooth_conv_in = ConvolutionTriplet_Modulated(input_channels, embed_dim, mod_dims=self.mod_dims, kernel_size=3) # Our input now has latent dimensions instead of 3 (RGB)
            self.patching = Patching_Modulated(embed_dim=embed_dim, patch_size=patch_size, mod_dims=self.mod_dims, norm_layer=norm_layer)
        else:
            self.patching = Patching_Modulated(embed_dim=input_channels, patch_size=patch_size, mod_dims=self.mod_dims, norm_layer=norm_layer)

        ################################################
        # ENCODER
        ################################################
        
        self.encoder : List[nn.Module] = []
        stage_block_id = 0
        # Encoder Swin Blocks
        for i_stage in range(len(depths)):

            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage

            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double
                stage.append(
                    SwinTransformerBlockV2_Modulated(
                        dim,
                        num_heads[i_stage],
                        mod_dims=self.mod_dims, 
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            self.encoder.append(ConditionedSequential(*stage))
            # Patch Merging Layer
            if i_stage < (len(depths) - 1) or self.final_downsample:
                self.encoder.append(PatchMergingV2_Modulated(dim, mod_dims=self.mod_dims, norm_layer=norm_layer))

        self.encoder = nn.ModuleList(self.encoder)

        middle_stage_features = embed_dim * 2 ** (len(depths) - int(not self.final_downsample))

        ################################################
        # MIDDLE
        ################################################

        self.set_positional_embedding(input_size)

        self.middle : List[nn.Module] = []
        for _ in range((global_stages)):
            self.middle.append(
                ViTEncoderBlock_Modulated(
                    num_heads = num_heads[-1], # Use number of heads as final Swin Encoder Block
                    hidden_dim = middle_stage_features,
                    mlp_dim = int(middle_stage_features * mlp_ratio), 
                    mod_dims=self.mod_dims,
                    dropout = dropout,
                    attention_dropout = attention_dropout,
                    norm_layer = norm_layer,
                )
            )
        self.middle = ConditionedSequential(*self.middle) if len(self.middle) > 0 else lambda x, c: x

        ################################################
        # DECODER
        ################################################

        self.decoder : List[nn.Module] = []

        # stage_block_id = 0 # NOTE : Not reseting dropout scheduler
        # Decoder Swin Blocks
        for i_stage in range(len(depths)-1, -1, -1): # NOTE : Count Backwards!
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage

            # add patch merging layer
            if i_stage < (len(depths) - 1) or self.final_downsample:
                self.decoder.append(PatchExpandingV2_Modulated(2*dim, mod_dims=self.mod_dims, norm_layer=norm_layer)) # NOTE : Double input dim

            if self.residual_cross_attention:
              self.decoder.append(
                SwinResidualCrossAttention_Modulated(window_size=window_size, embed_dim=dim, 
                                           num_heads=num_heads[i_stage], mod_dims=self.mod_dims, attention_dropout=attention_dropout,
                                           norm_layer=norm_layer))

            for i_layer in range(depths[i_stage]):
                # "Dropout Scheduler" : adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (2*total_stage_blocks - 1) # NOTE : Double

                stage.append(
                    SwinTransformerBlockV2_Modulated(
                        dim * (1 + int(i_layer == 0)), # First Swin Block in Decoder Stage gets Stacked from Residuals
                        num_heads[i_stage] * (1 + int(i_layer == 0)), # Double heads in this case too
                        mod_dims=self.mod_dims,
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                if i_layer == 0:
                    stage.append(
                        PointwiseConvolution_Modulated(2*dim, dim, mod_dims=self.mod_dims) # Reduce the dimensionality after first Swin in Stage
                    )
                stage_block_id += 1
            self.decoder.append(ConditionedSequential(*stage))

        self.decoder = nn.ModuleList(self.decoder)

        self.unpatching = UnPatching_Modulated(embed_dim=embed_dim, mod_dims=self.mod_dims, patch_size=patch_size) # Norm Layer uses default BatchNorm

        if self.smooth_conv:
            self.smooth_conv_out = ConvolutionTriplet_Modulated(2*embed_dim, embed_dim, mod_dims=self.mod_dims, kernel_size=3) # This will concatonate as a residual connection

        self.head = PointwiseConvolution_Modulated(embed_dim, output_channels, mod_dims=self.mod_dims, channel_last=False) # NOTE : we now "segment" back to the original dimensionality

        initialize_weights(self)

        initalize_diffusion(self)

        self.load(weights)

    def forward(self, x, t=None, y=None):
        """
        Forward pass.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # These defaults are for torchinfo.summary 
        if t is None:
            t = torch.zeros(x.size(0), dtype=torch.int, device=x.device)
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.int, device=x.device)

        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        original_spatial_shape = (x.size(-2), x.size(-1))

        if self.smooth_conv:
            conv_residual = self.smooth_conv_in(x, c)
            x = self.patching(conv_residual, c) # B C H W -> B H W C
        else:
            x = self.patching(x, c)
        
        residuals = []
        for i in range(0, len(self.encoder), 2):
        
            x = self.encoder[i](x, c) # Encoder Stage

            residuals.append(x)
            if i+1 < (len(self.encoder) - 1) or self.final_downsample:
                x = self.encoder[i+1](x, c) # Downsample (PatchMerge)

        if self.has_global_stages and not isinstance(self.middle, nn.Identity):
            *B, H, W, C = x.shape
            x = x.contiguous().view(*B, H*W, C)

            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.middle(x, c) # ViT Encoder

            x = x.contiguous().view(*B, H, W, C)

        for i_residual, i in zip(
            range(len(residuals)-1, -1, -1), # Count backwards for residual indices 
            range(0 - int(not self.final_downsample), len(self.decoder), 2 + int(self.residual_cross_attention))
        ):
            
            if i > 0 or self.final_downsample:
                residual_spatial_shape = residuals[i_residual].shape[-3:-1] # B H W C
                x = self.decoder[i](x, c, target_shape=residual_spatial_shape) # Upsample (PatchExpand)

            if self.residual_cross_attention:
                residual = self.decoder[i+1](x, residuals[i_residual], c) # Cross Attention Skip Connection
            else:
                residual = residuals[i_residual]

            x = torch.cat((x, residual), dim=-1) # Dumb Skip Connection

            x = self.decoder[i+(1 + int(self.residual_cross_attention))](x, c) # Decoder Stage


        # Does equally spaced padding to recover the original shape to concat with
        x = self.unpatching(x, c, target_shape=original_spatial_shape) # B H W C -> B C H W

        if self.smooth_conv:
            x = torch.cat((x, conv_residual), dim=-3) # ..., C, H, W
            x = self.smooth_conv_out(x, c)

        x = self.head(x, c)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
            """
            Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
            """
            # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1) 

    def set_positional_embedding(self, input_size):

        if input_size is None or not self.has_global_stages:
            self.pos_embed = None
            return None
        
        device = next(self.parameters()).device
        downsample_count = len(self.depths) - int(not self.final_downsample) # downsample is one less in this case

        latent_H = input_size[0] // self.patch_size[0]
        latent_W = input_size[1] // self.patch_size[1]
        for i in range(downsample_count):
            latent_H = (latent_H // 2) + (latent_H % 2) # Dims are padded up
            latent_W = (latent_W // 2)+ (latent_W % 2) # Dims are padded up

        middle_stage_features = self.embed_dim * 2 ** (len(self.depths) - int(not self.final_downsample))
        self.pos_embed = create_positional_embedding(middle_stage_features, latent_H, latent_W, device)

        return self.pos_embed
