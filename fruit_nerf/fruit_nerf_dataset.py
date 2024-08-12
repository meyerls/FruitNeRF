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
from dataclasses import dataclass
from typing import Optional, Union, Literal

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset,
)
from nerfstudio.utils.rich_utils import CONSOLE
from segmentation.grounded_sam.GroundingDINO.groundingdino.util.inference import Model
from segment_anything_hq import sam_model_registry, SamPredictor

import os
import json
import torch
import torchvision
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseImageSegmentation:
    def __init__(self, device, debug):
        self.device = device
        self.debug = debug

    def run(self, **kwargs):
        pass


class GroundedSAM(BaseImageSegmentation):
    def __init__(self, device='cpu', debug=False):
        super().__init__(device=device, debug=debug)

        import segmentation
        weights_base_path = Path(segmentation.__path__[0])

        self.model_config_path = weights_base_path / 'grounded_sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.model_checkpoint_path = weights_base_path / 'grounded_sam/groundingdino_swint_ogc.pth'
        self.sam_checkpoint = weights_base_path / 'grounded_sam/sam_vit_h_4b8939.pth'
        self.sam_hq_checkpoint_h = weights_base_path / 'grounded_sam/sam_hq_vit_h.pth'
        self.sam_hq_checkpoint_l = weights_base_path / 'grounded_sam/sam_hq_vit_l.pth'

        self.SAM_ENCODER_VERSION = "vit_h"

        if self.SAM_ENCODER_VERSION == "vit_h":
            self.sam_hq_checkpoint = self.sam_hq_checkpoint_h
        elif self.SAM_ENCODER_VERSION == "vit_l":
            self.sam_hq_checkpoint = self.sam_hq_checkpoint_l
        else:
            raise ValueError("Wrong checkpoint for SAM encoder")

        self.model = self.load_model(self.model_config_path, self.model_checkpoint_path, self.device)

    def load_model(self, model_config_path, model_checkpoint_path, device):
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.model_config_path,
                                          model_checkpoint_path=self.model_checkpoint_path)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.sam_hq_checkpoint)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def run(self,
            image_path: Union[Path, str],
            text_prompt: Optional[str],
            output_filename,
            output_dir: Union[Path, str],
            box_threshold: float = 0.35,
            text_threshold: float = 0.35,
            flag_segmentation_image_debug=False):

        # Define threshold for segmentation
        BOX_THRESHOLD = box_threshold  # Default was 0.35
        TEXT_THRESHOLD = text_threshold  # Default was 0.35
        NMS_THRESHOLD = 0.9  # Default was 0.8

        image = self.load_image(image_path.__str__())

        if isinstance(text_prompt, str):
            CLASSES = [text_prompt]
        elif isinstance(text_prompt, list):
            CLASSES = text_prompt
        else:
            raise ValueError("Text prompt is wrong: {}".format(text_prompt))

        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()

        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for confidence, class_id
            in zip(list(detections.confidence), list(detections.class_id))]

        # Display DINO bounding boxes
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # #save the annotated grounding dino image
        # cv2.imwrite(os.path.join(output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # Remove masks which are larger than 20% of the image
        for detection_idx in range(detections.__len__()):
            if detections[detection_idx].area > image.shape[0] * image.shape[1] * 0.2:
                # detections[detection_idx].mask = np.zeros_like(detections[detection_idx].mask, dtype=np.bool_)
                detections.mask[detection_idx] = np.zeros_like(detections[detection_idx].mask, dtype=np.bool_)
                detections.xyxy[detection_idx] = np.asarray([1, 0, 1, 0])

        # annotate image with detections
        mask_annotator = sv.MaskAnnotator()
        # annotated_image_rgb = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = mask_annotator.annotate(scene=np.zeros_like(image), detections=detections)

        if flag_segmentation_image_debug:
            box_annotator = sv.BoxAnnotator()
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        mask_path = os.path.join(output_dir, output_filename.name)
        mask = np.uint8(annotated_image.clip(0, 1).sum(axis=-1)) * 255
        cv2.imwrite(mask_path, mask)

        if flag_segmentation_image_debug:
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            annotated_image_path = os.path.join(output_dir, "overlay_" + output_filename.name)
            cv2.imwrite(annotated_image_path, annotated_image)

        return {"image": mask, "path": mask_path}


class SegmentImages(object):
    def __init__(self,
                 model: Literal['grounded_sam'] = 'grounded_sam',
                 device: Literal['cpu', 'cuda'] = 'cpu',
                 debug: bool = False):
        self.model_type = model

        if self.model_type == 'grounded_sam':
            self.model = GroundedSAM(device=device, debug=debug)
        else:
            raise ValueError("Type {} is not implemented".format(self.model_type))

    def run(self,
            image_path,
            text_prompt,
            output_filename,
            output_dir,
            box_threshold: float,
            text_threshold: float,
            flag_segmentation_image_debug: bool):
        return self.model.run(image_path,
                              text_prompt,
                              output_filename,
                              output_dir,
                              box_threshold,
                              text_threshold,
                              flag_segmentation_image_debug)


@dataclass
class FruitNerfDataset(ColmapConverterToNerfstudioDataset):
    """Process images into a nerfstudio dataset"""

    segmentation_class: str = None
    """Define text prompt/class to segment images. Currently grounded SAM is implemented"""
    text_threshold: float = 0.15
    """Threshold for text prompt/class to segment images. """
    box_threshold: float = 0.15
    """Threshold for bounding box prediction. """
    flag_segmentation_image_debug: bool = False
    """If True, also the masks overlay on rgb images will be saved."""
    data_semantic: Optional[Union[Path, str, bool]] = False
    """Define path to pre computed semantic masks."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""

    @property
    def semantic_dir(self) -> Path:
        return self.output_dir / "semantics"

    def main(self) -> None:
        """Process images into a nerfstudio dataset -> FruitNeRF Dataset."""

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
            else:  # Segment image data
                if "+" in self.segmentation_class:
                    # Concat class names to list
                    self.segmentation_class = [seg for seg in self.segmentation_class.split("+")]
                for image_path in image_rename_map_paths:
                    # Run image segmentation
                    mask = segmentor.run(image_path=image_path,
                                         text_prompt=self.segmentation_class,
                                         output_filename=image_rename_map_paths[image_path],
                                         output_dir=self.semantic_dir,
                                         text_threshold=self.text_threshold,
                                         box_threshold=self.box_threshold,
                                         flag_segmentation_image_debug=self.flag_segmentation_image_debug)
                    semantics_rename_map_paths[Path(image_path)] = Path(mask['path'])
                # Delete to free space on GPU
                del segmentor

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

        # COLMAP to  transform.json
        summary_log += self._save_transforms(
            num_frames,
            image_id_to_depth_path,
            None,
            image_rename_map=image_rename_map,
        )

        # Save semantic path to transform.json
        if self.segmentation_class or self.data_semantic:
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


if __name__ == '__main__':
    path = "/home/se86kimy/Dropbox/07_data/For5G/24_03_26/Drohne_4KK/DCIM/DJI_202403261424_005/Tree_row_4k_short_multiple_heights/result/images_2/frame_00018.png"
    gs = GroundedSAM()
    gs.run(image_path=path, output_dir="./test", text_prompt=['tree'], output_filename=Path('test_tree.png'), flag_segmentation_image_debug=True)
