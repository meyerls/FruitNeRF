<h1 style="text-align: center;">:apple: :pear: :peach: :lemon:</h1>
<h1 style="text-align: center;">FruitNeRF:  A Generalized Framework for Counting Fruits in Neural Radiance Fields </h1>

**Abstract**:
We introduce FruitNeRF, a unified novel fruit counting framework that leverages state-of-the-art view synthesis methods
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
mangoes. Additionally, we assess the performance of fruit counting using the foundation model compared to a U-Net.

:globe_with_meridians:[[ Project Page]](https://meyerls.github.io/fruit_nerf/):page_facing_up:[[ Paper]](https://meyerls.github.io/fruit_nerf/):file_folder:[[ Dataset]](https://meyerls.github.io/fruit_nerf/)


# Note
 
Code will be published soon!