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

# !/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""

from typing import Union

import tyro
from typing_extensions import Annotated

from fruit_nerf.fruit_nerf_dataset import (
    FruitNerfDataset,
)

from nerfstudio.utils.rich_utils import CONSOLE

Commands = Union[
    Annotated[FruitNerfDataset, tyro.conf.subcommand(name="fruit")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")

    #try:
    tyro.cli(Commands).main()
    #except RuntimeError as e:
    #    CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()

def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # type: ignore
