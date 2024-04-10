"""
FruitNeRF implementation .
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import JaccardIndex

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
from nerfstudio.field_components.encodings import NeRFEncoding

from fruit_nerf.fruit_field import FruitField, SemanticNeRFField
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
    """FruitModel based on Nerfacto model

    Args:
        config: FruitModel configuration to instantiate model
    """

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

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

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
        # Build the proposal network(s)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
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
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Samplers
        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.test_mode == 'inference':
            self.num_inference_samples = None  # int(200)
            self.proposal_sampler = None  # UniformSamplerWithNoise(num_samples=self.num_inference_samples, single_jitter=True)
            self.field.spatial_distortion = None
        elif self.config.proposal_initial_sampler == "uniform":
            # Change proposal network initial sampler if uniform
            initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
            if self.config.proposal_initial_sampler == "uniform":
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

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()
        # Include this renderer in FruitNeRF:
        # class SemanticRenderer(nn.Module):
        #    """Calculate semantics along the ray."""

        #    @classmethod
        #    def forward(
        #            cls,
        #            semantics: Float[Tensor, "*bs num_samples num_classes"],
        #            weights: Float[Tensor, "*bs num_samples 1"],
        #    ) -> Float[Tensor, "*bs num_classes"]:
        #        """Calculate semantics along the ray."""
        #        sem = torch.sum(weights * semantics, dim=-2)
        #        return sem

        # losses
        self.rgb_loss = MSELoss()
        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def setup_inference(self, render_rgb, num_inference_samples):
        self.render_rgb = render_rgb  # True
        self.num_inference_samples = num_inference_samples  # int(200)
        self.proposal_sampler = UniformSamplerWithNoise(num_samples=self.num_inference_samples, single_jitter=False)
        self.field.spatial_distortion = None

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_inference_outputs(self, ray_bundle: RayBundle, render_rgb: bool = False):
        outputs = {}

        ray_samples = self.proposal_sampler(ray_bundle)
        field_outputs = self.field.forward(ray_samples, render_rgb=render_rgb)

        if render_rgb:
            outputs["rgb"] = field_outputs[FieldHeadNames.RGB]

        outputs['point_location'] = ray_samples.frustums.get_positions()
        outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]

        semantic_labels = torch.sigmoid(outputs["semantics"])
        threshold = 0.9
        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)

        outputs["semantics_colormap"] = semantic_labels

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):  #

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
        }

        # depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # accumulation = self.renderer_accumulation(weights=weights)

        # outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

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

        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        # semantic loss
        # batch["fruit_mask"][batch["fruit_mask"] < 0.1] = 0
        # batch["fruit_mask"][batch["fruit_mask"] >= 0.1] = 1

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
            fruit_nerf_output = self.get_inference_outputs(ray_bundle, True)
        else:
            fruit_nerf_output = self.get_outputs(ray_bundle)

        return fruit_nerf_output

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
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
        # semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        semantic_labels = torch.sigmoid(outputs["semantics"])
        images_dict[
            "semantics_colormap"] = semantic_labels  # colormaps.apply_float_colormap(semantic_labels).cpu().numpy()  #self.colormap.to(self.device)[semantic_labels]

        # valid mask
        images_dict["fruit_mask"] = batch["fruit_mask"].repeat(1, 1, 3).to(self.device)
        #batch["fruit_mask"][batch["fruit_mask"] < 0.1] = 0
        #batch["fruit_mask"][batch["fruit_mask"] >= 0.1] = 1

        from torchmetrics.classification import BinaryJaccardIndex
        metric = BinaryJaccardIndex().to(self.device)
        semantic_labels = torch.nn.functional.softmax(outputs["semantics"])
        iou = metric(semantic_labels[..., 0], batch["fruit_mask"][..., 0])
        metrics_dict["iou"] = float(iou)

        return metrics_dict, images_dict


# from nerfstudio.fields.vanilla_nerf_field import NeRFField
# from nerfstudio.utils import colormaps, colors, misc
# from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
# from nerfstudio.configs.config_utils import to_immutable_dict
# from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
# from typing import Any, Dict, List, Tuple, Type
# from nerfstudio.models.base_model import Model, ModelConfig
#
# @dataclass
# class FruitNerfMLPModelConfig(ModelConfig):
#    """Nerfacto Model Config"""
#
#    _target: Type = field(default_factory=lambda: FruitModelMLP)
#    semantic_loss_weight: float = 1.0
#    pass_semantic_gradients: bool = False
#    num_coarse_samples: int = 64
#    """Number of samples in coarse field evaluation"""
#    num_importance_samples: int = 128
#    """Number of samples in fine field evaluation"""
#
#    enable_temporal_distortion: bool = False
#    """Specifies whether or not to include ray warping based on time."""
#    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
#    """Parameters to instantiate temporal distortion with"""
#
#
class FruitModelMLP(Model):
    pass
#    """Vanilla model
#
#    Args:
#        config: Nerfacto configuration to instantiate model
#    """
#
#    config: FruitNerfMLPModelConfig
#
#    def __init__(self, config: FruitNerfModelConfig, metadata: Dict, **kwargs) -> None:
#        assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
#        self.semantics = metadata["semantics"]
#        self.test_mode = kwargs['test_mode']
#        self.render_rgb_inference = kwargs['render_rgb_inference']
#
#        self.field_coarse = None
#        self.field_fine = None
#        self.temporal_distortion = None
#
#        super().__init__(config=config, **kwargs)
#        self.colormap = self.semantics.colors.clone().detach().to(self.device)
#
#    def populate_modules(self):
#        """Set the fields and modules."""
#        super().populate_modules()
#
#        # Fields
#        position_encoding = NeRFEncoding(
#            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
#        )
#        direction_encoding = NeRFEncoding(
#            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
#        )
#
#        self.field_coarse = SemanticNeRFField(
#            position_encoding=position_encoding,
#            direction_encoding=direction_encoding,
#            base_mlp_layer_width=196,
#            pass_semantic_gradients=self.config.pass_semantic_gradients,
#
#        )
#
#        self.field_fine = SemanticNeRFField(
#            position_encoding=position_encoding,
#            direction_encoding=direction_encoding,
#            base_mlp_layer_width=196,
#            pass_semantic_gradients=self.config.pass_semantic_gradients,
#        )
#
#        # samplers
#        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
#        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
#
#        # renderers
#        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
#        self.renderer_accumulation = AccumulationRenderer()
#        self.renderer_depth = DepthRenderer()
#        self.renderer_semantics = SemanticRenderer()
#
#        # losses
#        self.rgb_loss = MSELoss()
#        self.binary_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
#
#        # metrics
#        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
#        self.ssim = structural_similarity_index_measure
#        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
#
#        if getattr(self.config, "enable_temporal_distortion", False):
#            params = self.config.temporal_distortion_params
#            kind = params.pop("kind")
#            self.temporal_distortion = kind.to_temporal_distortion(params)
#
#    def get_param_groups(self) -> Dict[str, List[Parameter]]:
#        param_groups = {}
#        if self.field_coarse is None or self.field_fine is None:
#            raise ValueError("populate_fields() must be called before get_param_groups")
#        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
#        if self.temporal_distortion is not None:
#            param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
#        return param_groups
#
#    def get_inference_outputs(self, ray_bundle: RayBundle, render_rgb: bool = False):
#        outputs = {}
#
#        ray_samples = self.proposal_sampler(ray_bundle)
#        field_outputs = self.field_fine.forward(ray_samples)
#
#        if render_rgb:
#            outputs["rgb"] = field_outputs[FieldHeadNames.RGB]
#
#        outputs['point_location'] = ray_samples.frustums.get_positions()
#        outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
#        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]
#
#        semantic_labels = torch.sigmoid(outputs["semantics"])
#        threshold = 0.9
#        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
#
#        outputs["semantics_colormap"] = semantic_labels
#
#        return outputs
#
#    def get_outputs(self, ray_bundle: RayBundle):
#        if self.field_coarse is None or self.field_fine is None:
#            raise ValueError("populate_fields() must be called before get_outputs")
#
#        # uniform sampling
#        ray_samples_uniform = self.sampler_uniform(ray_bundle)
#        if self.temporal_distortion is not None:
#            offsets = None
#            if ray_samples_uniform.times is not None:
#                offsets = self.temporal_distortion(
#                    ray_samples_uniform.frustums.get_positions(), ray_samples_uniform.times
#                )
#            ray_samples_uniform.frustums.set_offsets(offsets)
#
#        # coarse field:
#        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
#        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
#        rgb_coarse = self.renderer_rgb(
#            rgb=field_outputs_coarse[FieldHeadNames.RGB],
#            weights=weights_coarse,
#        )
#        accumulation_coarse = self.renderer_accumulation(weights_coarse)
#        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)
#
#        # pdf sampling
#        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
#        if self.temporal_distortion is not None:
#            offsets = None
#            if ray_samples_pdf.times is not None:
#                offsets = self.temporal_distortion(ray_samples_pdf.frustums.get_positions(), ray_samples_pdf.times)
#            ray_samples_pdf.frustums.set_offsets(offsets)
#
#        # fine field:
#        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
#        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
#        rgb_fine = self.renderer_rgb(
#            rgb=field_outputs_fine[FieldHeadNames.RGB],
#            weights=weights_fine,
#        )
#        accumulation_fine = self.renderer_accumulation(weights_fine)
#        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
#
#        outputs = {
#            "rgb_coarse": rgb_coarse,
#            "rgb_fine": rgb_fine,
#            "accumulation_coarse": accumulation_coarse,
#            "accumulation_fine": accumulation_fine,
#            "depth_coarse": depth_coarse,
#            "depth_fine": depth_fine,
#        }
#
#        # semantics
#        semantic_weights = weights_fine
#        if not self.config.pass_semantic_gradients:
#            semantic_weights = semantic_weights.detach()
#        outputs["semantics"] = self.renderer_semantics(
#            field_outputs_fine[FieldHeadNames.SEMANTICS], weights=semantic_weights
#        )
#
#        # semantics colormaps
#        semantic_labels = torch.sigmoid(outputs["semantics"].detach())
#        threshold = 0.9
#        semantic_labels = torch.heaviside(semantic_labels - threshold, torch.tensor(0.)).to(torch.long)
#
#        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]
#
#        return outputs
#
#    def setup_inference(self, render_rgb, num_inference_samples):
#        self.render_rgb = render_rgb  # True
#        self.num_inference_samples = num_inference_samples  # int(200)
#        self.proposal_sampler = UniformSamplerWithNoise(num_samples=self.num_inference_samples, single_jitter=False)
#
#    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
#        # Scaling metrics by coefficients to create the losses.
#        device = outputs["rgb_coarse"].device
#        image = batch["image"].to(device)
#
#        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
#        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
#
#        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
#        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
#
#        # semantic loss
#        loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.binary_cross_entropy_loss(
#            outputs["semantics"], batch["fruit_mask"])
#
#        return loss_dict
#
#    def get_image_metrics_and_images(
#            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
#    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
#        image = batch["image"].to(outputs["rgb_coarse"].device)
#        image = self.renderer_rgb.blend_background(image)
#        rgb_coarse = outputs["rgb_coarse"]
#        rgb_fine = outputs["rgb_fine"]
#        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
#        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
#        assert self.config.collider_params is not None
#        depth_coarse = colormaps.apply_depth_colormap(
#            outputs["depth_coarse"],
#            accumulation=outputs["accumulation_coarse"],
#            near_plane=self.config.collider_params["near_plane"],
#            far_plane=self.config.collider_params["far_plane"],
#        )
#        depth_fine = colormaps.apply_depth_colormap(
#            outputs["depth_fine"],
#            accumulation=outputs["accumulation_fine"],
#            near_plane=self.config.collider_params["near_plane"],
#            far_plane=self.config.collider_params["far_plane"],
#        )
#
#        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
#        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
#        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)
#
#        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
#        image = torch.moveaxis(image, -1, 0)[None, ...]
#        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
#        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
#
#        coarse_psnr = self.psnr(image, rgb_coarse)
#        fine_psnr = self.psnr(image, rgb_fine)
#        fine_ssim = self.ssim(image, rgb_fine)
#        fine_lpips = self.lpips(image, rgb_fine)
#        assert isinstance(fine_ssim, torch.Tensor)
#
#        metrics_dict = {
#            "psnr": float(fine_psnr.item()),
#            "coarse_psnr": float(coarse_psnr),
#            "fine_psnr": float(fine_psnr),
#            "fine_ssim": float(fine_ssim),
#            "fine_lpips": float(fine_lpips),
#        }
#        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
#        return metrics_dict, images_dict
#
#    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
#        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
#        of the model and whether or not the batch is provided (whether or not we are training basically)
#
#        Args:
#            ray_bundle: containing all the information needed to render that ray latents included
#        """
#
#        if self.collider is not None:
#            ray_bundle = self.collider(ray_bundle)
#
#        if self.test_mode == 'inference':
#            fruit_nerf_output = self.get_inference_outputs(ray_bundle, self.render_rgb)
#        else:
#            fruit_nerf_output = self.get_outputs(ray_bundle)
#
#        return fruit_nerf_output
