"""
FruitNeRF implementation.
"""

from __future__ import annotations
from collections import defaultdict

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.semantic_nerf_field import SemanticNerfField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.models.nerfacto import NerfactoModelConfig

from fruit_nerf.fruit_field import FruitField
from fruit_nerf.components.ray_samplers import UniformSamplerWithNoise


@dataclass
class FruitNerfModelConfig(NerfactoModelConfig):
    """FruitModel Model Config"""

    _target: Type = field(default_factory=lambda: FruitModel)
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False
    num_layers_semantic: int = 2
    hidden_dim_semantics: int = 64
    geo_feat_dim: int = 15


class FruitModel(Model):
    """FruitModel based on Nerfacto model"""

    config: FruitNerfModelConfig

    def __init__(self, config: FruitNerfModelConfig, metadata: Dict, **kwargs) -> None:
        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
        self.semantics = metadata["semantics"]
        self.test_mode = kwargs['test_mode']
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf")) if not self.config.disable_scene_contraction else None

        # Fields
        self.field = FruitField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            num_layers_semantic=self.config.num_layers_semantic,
            hidden_dim_semantics=self.config.hidden_dim_semantics,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            geo_feat_dim=self.config.geo_feat_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_semantics=True,
            test_mode=self.test_mode,
            num_semantic_classes=1,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        self.density_fns = []
        self.proposal_networks = torch.nn.ModuleList()
        num_prop_nets = self.config.num_proposal_iterations

        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
                self.density_fns.append(network.density_fn)

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )


        initial_sampler = None
        if self.test_mode == 'inference':
            self.num_inference_samples = None
            self.proposal_sampler = None
            self.field.spatial_distortion = None
        elif self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        else:
            self.proposal_sampler = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                single_jitter=self.config.use_single_jitter,
                update_sched=update_schedule,
                initial_sampler=initial_sampler,
            )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # Losses
        self.rgb_loss = MSELoss()
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters())
        }

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output.cpu())
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_export_outputs(self, ray_bundle: RayBundle):
        outputs = {}
        ray_samples = self.proposal_sampler(ray_bundle)
        field_outputs = self.field.forward(ray_samples)

        outputs["rgb"] = field_outputs[FieldHeadNames.RGB]

        outputs["point_location"] = ray_samples.frustums.get_positions()
        outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]

        semantic_labels = torch.sigmoid(outputs["semantics"])
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = semantic_labels

        return outputs

    def get_inference_outputs(self, ray_bundle: RayBundle):
        outputs = {}


        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb,
                   "accumulation": accumulation,
                   "depth": depth,
                   "weights_list": weights_list,
                   "ray_samples_list": ray_samples_list}

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # semantics
        semantic_weights = weights
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels].repeat(1, 3)

        return outputs


    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)


        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list
        }

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # Semantics
        semantic_weights = weights if self.config.pass_semantic_gradients else weights.detach()
        outputs["semantics"] = self.renderer_semantics(field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights)

        # Semantics colormaps
        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        
        # Ensure outputs_rgb is always defined
        outputs_rgb = outputs["rgb"]
        
        # Handle channel mismatch
        if outputs["rgb"].shape[-1] != image.shape[-1]:
            if outputs["rgb"].shape[-1] == 4 and image.shape[-1] == 3:
                outputs_rgb = outputs["rgb"][..., :3]  # Use only the RGB channels from the output
            elif outputs["rgb"].shape[-1] == 3 and image.shape[-1] == 4:
                image = image[..., :3]  # Use only the RGB channels from the ground truth
            else:
                raise ValueError(f"Unexpected channel size in tensors: outputs['rgb'] shape {outputs['rgb'].shape}, image shape {image.shape}")
        
        # Calculate RGB loss
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs_rgb)
        
        # Other loss calculations can go here
        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
            outputs["semantics"], batch["fruit_mask"]
        )
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        
        return loss_dict

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        if self.test_mode == 'inference':
            # fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
            fruit_nerf_output = self.get_inference_outputs(ray_bundle)
        elif self.test_mode == 'export':
            # fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
            fruit_nerf_output = self.get_export_outputs(ray_bundle)
        else:
            fruit_nerf_output = self.get_outputs(ray_bundle)

        return fruit_nerf_output

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        
        # Ensure outputs_rgb is always defined
        outputs_rgb = outputs["rgb"]
        
        # Handle channel mismatch
        if outputs["rgb"].shape[-1] != image.shape[-1]:
            if outputs["rgb"].shape[-1] == 4 and image.shape[-1] == 3:
                outputs_rgb = outputs["rgb"][..., :3]  # Use only the RGB channels from the output
            elif outputs["rgb"].shape[-1] == 3 and image.shape[-1] == 4:
                image = image[..., :3]  # Use only the RGB channels from the ground truth
            else:
                raise ValueError(f"Unexpected channel size in tensors: outputs['rgb'] shape {outputs['rgb'].shape}, image shape {image.shape}")
        
        # Compute PSNR
        metrics_dict["psnr"] = self.psnr(outputs_rgb, image)
        
        # Compute distortion loss
        metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        
        return metrics_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        # semantics
        semantic_labels = torch.sigmoid(outputs["semantics"])
        images_dict["semantics_colormap"] = semantic_labels

        # valid mask
        images_dict["fruit_mask"] = batch["fruit_mask"].repeat(1, 1, 3).to(self.device)

        from torchmetrics.classification import BinaryJaccardIndex
        metric = BinaryJaccardIndex().to(self.device)
        semantic_labels = torch.nn.functional.softmax(outputs["semantics"])
        iou = metric(semantic_labels[..., 0], batch["fruit_mask"][..., 0])
        metrics_dict["iou"] = float(iou)

        return metrics_dict, images_dict
