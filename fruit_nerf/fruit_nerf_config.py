"""
FruitNeRF configuration file.
"""

from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

# FruitNerf
from fruit_nerf.fruit_pipeline import FruitPipeline, FruitPipelineConfig
from fruit_nerf.fruit_nerf import FruitNerfModelConfig  # , FruitNerfMLPModelConfig
from fruit_nerf.data.fruit_dataset import FruitDataset
from fruit_nerf.data.fruitnerf_dataparser import FruitNerfDataParserConfig
from fruit_nerf.data.fruit_datamanager import FruitDataManager, FruitDataManagerConfig

# FruitNerf
fruit_nerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="fruit_nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=FruitPipelineConfig(
            datamanager=FruitDataManagerConfig(
                dataparser=FruitNerfDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=FruitNerfModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for FruitNeRF",
)

fruit_nerf_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="fruit_nerf_big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=FruitPipelineConfig(
            datamanager=FruitDataManagerConfig(
                train_num_images_to_sample_from=200,
                train_num_times_to_repeat_images=1000,
                dataparser=FruitNerfDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096 * 2,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3),
                ),
            ),
            model=FruitNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                geo_feat_dim=30,
                hidden_dim_color=128,
                hidden_dim_semantics=128,
                num_layers_semantic=3,
                appearance_embed_dim=128,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for FruitNeRF-Big",
)

fruit_nerf_method_huge = MethodSpecification(
    config=TrainerConfig(
        method_name="fruit_nerf_huge",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=FruitPipelineConfig(
            datamanager=FruitDataManagerConfig(
                dataparser=FruitNerfDataParserConfig(),
                train_num_rays_per_batch=4096 * 4,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-5, max_steps=50000),
                ),
            ),
            model=FruitNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(512, 512),
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
                ],
                hidden_dim=256,
                hidden_dim_color=256,
                appearance_embed_dim=32,
                geo_feat_dim=30,
                hidden_dim_semantics=128,
                num_layers_semantic=3,
                max_res=8192,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for FruitNeRF-Huge",
)
