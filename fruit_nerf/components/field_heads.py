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

"""
Collection of render heads
"""
from enum import Enum
from typing import Callable, Optional, Union

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.field_heads import FieldHead, FieldHeadNames


class SemanticFieldHead(FieldHead):
    """Semantic output

    Args:
        num_classes: Number of semantic classes
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, num_classes: int, in_dim: Optional[int] = None, activation=None) -> None:
        super().__init__(in_dim=in_dim, out_dim=num_classes, field_head_name=FieldHeadNames.SEMANTICS,
                         activation=activation)
