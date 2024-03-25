<h1 style="text-align: center;">:apple: :pear: FruitNeRF:  A Generalized Framework for Counting Fruits in Neural Radiance Fields :peach: :lemon:</h1>


<p style="align:justify"><b>Abstract</b>: We introduce FruitNeRF, a unified novel fruit counting framework that leverages state-of-the-art view synthesis methods
to count any fruit type directly in 3D. Our framework takes an unordered set of posed images captured by a monocular
camera and segments fruit in each image. To make our system independent of the fruit type, we employ a foundation model
that generates binary segmentation masks for any fruit. Utilizing both modalities, RGB and semantic, we train a semantic
neural radiance field. Through uniform volume sampling of the implicit Fruit Field, we obtain fruit-only point clouds.
By applying cascaded clustering on the extracted point cloud, our approach achieves precise fruit count. The use of
neural radiance fields provides significant advantages over conventional methods such as object tracking or optical
flow, as the counting itself is lifted into 3D. Our method prevents double counting fruit and avoids counting irrelevant
fruit. We evaluate our methodology using both real-world and synthetic datasets. The real-world dataset consists of
three apple trees with manually counted ground truths, a benchmark apple dataset with one row and ground truth fruit
location, while the synthetic dataset comprises various fruit types including apple, plum, lemon, pear, peach, and
mangoes. Additionally, we assess the performance of fruit counting using the foundation model compared to a U-Net.</p>

<p align="center">:globe_with_meridians:[[Project Page]](https://meyerls.github.io/fruit_nerf/)  :page_facing_up:[[ Paper]](https://meyerls.github.io/fruit_nerf/)  :file_folder:[[ Dataset]](https://meyerls.github.io/fruit_nerf/) </p>

# Installation

### 0. Install Nerfstudio dependencies

[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "
tinycudann" to install dependencies and create an environment

### 1. Clone this repo

`git clone https://github.com/meyerls/fruit_nerf`

### 2. Install this repo as a python package

Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install

Run `ns-train -h`: you should see a list of "subcommand" with fruit_nerf included among them.

# Using FruitNeRF

Now that FruitNeRF is installed you can start counting fruits!

## Prepare own Data

```bash
ns-prepocess-fruit-data ...
```

## Training

```bash
ns-train fruit_nerf --data {path/to/image-dir}
```

## Volumetric Sampling

```bash
ns-export-semantics semantic-pointcloud --load-config {path/to/config.yaml} --output-dir {path/to/export/dir} --use-bounding-box True --bounding-box-min -0.2 -0.2 -0.26 --bounding-box-max 0.2 0.2 0.05 --num_rays_per_batch 2000 --num_points_per_side 1000
```

## Point Cloud Clustering / Fruit Counting

```bash
ns-fruits count --data {path/to/semantic-point-cloud}
```

## Bibtex

If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{fruitnerf2024,
&nbsp;author = { Lukas Meyer, Andreas Gilson, Ute Schmidt, Marc Stamminger},
&nbsp;title = {FruitNeRF: A Unified Neural Radiance Field based Fruit Counting Framework},
&nbsp;booktitle = {ArXiv},
&nbsp;year = {2024},
} </pre>