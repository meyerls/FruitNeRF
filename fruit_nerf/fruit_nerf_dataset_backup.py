# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes an image sequence to a nerfstudio compatible dataset."""
import os
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import os

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset
)
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import (
    BaseConverterToNerfstudioDataset,
)
from nerfstudio.utils.rich_utils import CONSOLE

import argparse
import os
import copy
import sys
import shutil
import subprocess

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

if False:
    # Grounding DINO
    if os.uname()[1] == 'austro':
        sys.path.append('/home/se86kimy/Documents/code/FruitNeRF/segmentation')
    else:
        sys.path.append('/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/nerfstudio/segmentation')
    import grounded_segment_anything.GroundingDINO.groundingdino.datasets.transforms as T
    from grounded_segment_anything.GroundingDINO.groundingdino.models import build_model
    from grounded_segment_anything.GroundingDINO.groundingdino.util import box_ops
    from grounded_segment_anything.GroundingDINO.groundingdino.util.slconfig import SLConfig
    from grounded_segment_anything.GroundingDINO.groundingdino.util.utils import clean_state_dict, \
        get_phrases_from_posmap

    # segment anything
    from segment_anything import (
        build_sam,
        build_sam_hq,
        SamPredictor
    )
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path

from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command


class BaseImageSemntation:
    def __init__(self, device, debug):
        self.device = device
        self.debug = debug

    def run(self, **kwargs):
        pass


class GroundedSAM(BaseImageSemntation):
    def __init__(self, device='cpu', debug=False):
        super().__init__(device=device, debug=debug)
        self.model_config_path = '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/nerfstudio/segmentation/grounded_segment_anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.model_checkpoint_path = '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/nerfstudio/segmentation/grounded_segment_anything/groundingdino_swint_ogc.pth'
        self.sam_hq_checkpoint = '/home/se86kimy/Dropbox/05_productive/01_code/07_NeRF/04_Nerfstudio/nerfstudio/segmentation/grounded_segment_anything/sam_hq_vit_h.pth'
        self.use_sam_hq = True

        self.model = self.load_model(self.model_config_path, self.model_checkpoint_path, self.device)

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True,
                             device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def run(self, image_path, text_prompt, output_filename, output_dir):
        image_pil, image = self.load_image(image_path)
        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.model, image, text_prompt, box_threshold=0.3, text_threshold=0.25, device=self.device
        )

        # initialize SAM
        if self.use_sam_hq:
            predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))

        image = cv2.imread(image_path.__str__())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)
            plt.axis('off')
            plt.show()

        value = 0  # 0 for background

        mask_img = torch.zeros(masks.shape[-2:])
        for idx, mask in enumerate(masks):
            if idx == 0:
                mask_img[mask.cpu().numpy()[0] == True] = 0
            else:
                mask_img[mask.cpu().numpy()[0] == True] = 255

        mask_path = os.path.join(output_dir, output_filename.name)
        cv2.imwrite(mask_path, mask_img.numpy())

        return {"image": mask_img.numpy(), "path": mask_path}


class SegmentImages(object):
    def __init__(self, model='grounded_sam', device='cpu', debug=False):
        self.model_type = model

        if self.model_type == 'grounded_sam':
            self.model = GroundedSAM(device=device, debug=debug)
        else:
            raise ValueError("Type {} is not implemented".format(self.model_type))

    def run(self, image_path, text_prompt, output_filename, output_dir):
        return self.model.run(image_path, text_prompt, output_filename, output_dir)


@dataclass
class FruitNerfDataset(ColmapConverterToNerfstudioDataset):
    """Process images into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """
    segmentation_class: str = 'apple'
    """Define text prompt/class to segment images. Currently grounded SAM is implemented"""
    data_semantic: Optional[Union[Path, str, bool]] = False
    """Define path to pre computed semantic masks."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        if self.segmentation_class:
            segmentor = SegmentImages(model='grounded_sam', device='cuda')

        if isinstance(self.data_semantic, str):
            self.data_semantic = Path(self.data_semantic)

        require_cameras_exist = False
        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if not self.skip_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
            require_cameras_exist = True

        image_rename_map: Optional[dict[str, str]] = None
        semantics_rename_map_paths: OrderedDict = OrderedDict()

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            pers_size = equirect_utils.compute_resolution_from_equirect(self.data, self.images_per_equirect)
            CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
            self.data = equirect_utils.generate_planar_projections_from_equirectangular(
                self.data, pers_size, self.images_per_equirect, crop_factor=self.crop_factor
            )
            self.camera_type = "perspective"

        summary_log = []

        # Copy and downscale images
        if not self.skip_image_processing:
            # Copy images to output directory
            image_rename_map_paths = process_data_utils.copy_images(
                self.data, image_dir=self.image_dir, crop_factor=self.crop_factor, verbose=self.verbose
            )
            image_rename_map = dict((a.name, b.name) for a, b in image_rename_map_paths.items())
            num_frames = len(image_rename_map)
            summary_log.append(f"Starting with {num_frames} images")

            # Downscale images
            summary_log.append(
                process_data_utils.downscale_images(self.image_dir, self.num_downscales, verbose=self.verbose)
            )

            os.makedirs(self.semantic_dir, exist_ok=True)

            # Copy Data
            if isinstance(self.data_semantic, Path):
                semantics_rename_map_paths = process_data_utils.copy_images(
                    self.data_semantic, image_dir=self.semantic_dir, crop_factor=self.crop_factor, verbose=self.verbose
                )
            # Segment image data
            else:
                for image_path in image_rename_map_paths:
                    mask = segmentor.run(image_path=image_path,
                                         text_prompt=self.segmentation_class,
                                         output_filename=image_rename_map_paths[image_path],
                                         output_dir=self.semantic_dir)
                    semantics_rename_map_paths[Path(image_path)] = Path(mask['path'])
                del (segmentor)

            semantic_rename_map = dict((a.name, b.name) for a, b in semantics_rename_map_paths.items())
            num_semantic_frames = len(semantic_rename_map)
            summary_log.append(f"Starting with {num_semantic_frames} semantic images")

            # Downscale images
            summary_log.append(
                process_data_utils.downscale_images(self.semantic_dir, self.num_downscales,
                                                    folder_name=self.semantic_dir.name,
                                                    verbose=self.verbose)
            )
        else:
            num_frames = len(process_data_utils.list_images(self.data))
            if num_frames == 0:
                raise RuntimeError("No usable images in the data folder.")
            summary_log.append(f"Starting with {num_frames} images")

        # Run COLMAP
        if not self.skip_colmap:
            require_cameras_exist = True
            self._run_colmap()
            # Colmap uses renamed images
            image_rename_map = None

        # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp

        if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
            raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")

        summary_log += self._save_transforms(
            num_frames,
            image_id_to_depth_path,
            None,
            image_rename_map=image_rename_map,
        )

        if self.segmentation_class:
            with open(os.path.join(self.output_dir, 'transforms.json')) as f:
                transform_json = json.load(f)
                transform_json.update({'semantics': ['stuff', self.segmentation_class]})
                for frame in transform_json['frames']:
                    frame.update({'semantic_path': os.path.join(self.semantic_dir.name, Path(frame['file_path']).name)})

            with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
                json.dump(transform_json, f, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)


def list_images(data: Path) -> List[Path]:
    """Lists all supported images in a directory

    Args:
        data: Path to the directory of images.
    Returns:
        Paths to images contained in the directory
    """
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    image_paths = sorted([p for p in data.glob("[!.]*") if p.suffix.lower() in allowed_exts])
    return image_paths


def downscale_images(
        orig_file: Path,
        new_file: Path,
        downscale_factor: int,
        nearest_neighbor: bool = False,
        verbose: bool = False,
):

    assert isinstance(downscale_factor, int)

    filename = orig_file
    nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
    ffmpeg_cmd = [
        f'ffmpeg -y -noautorotate -i "{filename}" ',
        f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
        f'"{new_file}"',
    ]
    ffmpeg_cmd = " ".join(ffmpeg_cmd)
    run_command(ffmpeg_cmd, verbose=verbose)


@dataclass
class FruitNerfSyntheticDataset(object):
    """Process images into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """
    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    verbose: bool = False
    """If True, print extra logging."""
    image_sets: Tuple[int, int, int, int] = (10, 20, 50, 100, 200, 300)
    """List of target image size of one dataset."""

    image_downscale: Tuple[int, int, int, int] = (1, 2, 4)
    """List of target image resolution of one dataset."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        with open(self.data / "transforms.json") as f:
            transform_main = json.load(f)

        assert transform_main['w'] == transform_main['h']

        image_resolution = [int(transform_main['w'] / x) for x in self.image_downscale]

        for image_res, image_downscale in zip(image_resolution, self.image_downscale):
            for image_size in self.image_sets:
                output_res = self.output_dir.parts[-1] + '_{}x{}'.format(image_res, image_res)
                output_dir_new = self.output_dir / output_res / (self.output_dir.parts[-1] + '_#' + str(image_size))
                output_dir_new.mkdir(exist_ok=True, parents=True)

                new_image_folder = output_dir_new / 'images'
                new_image_folder.mkdir(exist_ok=True)

                new_semantic_folder = output_dir_new / 'semantics'
                new_semantic_folder.mkdir(exist_ok=True)

                transform_new = copy.copy(transform_main)
                transform_new['frames'] = transform_new['frames'][:image_size]
                transform_new_path = output_dir_new / 'transforms.json'

                with open(str(transform_new_path), 'w', encoding='utf-8') as f:
                    json.dump(transform_new, f, ensure_ascii=False, indent=4)

                for frame in transform_new['frames']:
                    orig_image_file_path = self.data / frame['file_path']
                    new_image_file_path = output_dir_new / frame['file_path']
                    # shutil.copy(orig_image_file_path, new_image_file_path)
                    downscale_images(orig_file=orig_image_file_path,
                                     new_file=new_image_file_path,
                                     downscale_factor=image_downscale,
                                     verbose=self.verbose)

                    orig_semantic_file_path = self.data / frame['semantic_path']
                    new_semantic_file_path = output_dir_new / frame['semantic_path']
                    # shutil.copy(orig_semantic_file_path, new_semantic_file_path)
                    downscale_images(orig_file=orig_semantic_file_path,
                                     new_file=new_semantic_file_path,
                                     downscale_factor=image_downscale,
                                     verbose=self.verbose)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
