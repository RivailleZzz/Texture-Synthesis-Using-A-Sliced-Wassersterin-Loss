# Texture Synthesis Using A Sliced Wassersterin Loss

This is the official implementation of "A Sliced Wasserstein Loss for Neural Texture Synthesis" paper (to appear in CVPR 2021).

![image](https://user-images.githubusercontent.com/97613092/164089466-5b4b8796-419c-44fa-a825-bb02d046859b.png)

Run the Colab notebooks:

-style_transfer.ipynb

-texture_synthesis.ipynb

-texture_synthesis_with_tags.ipynb

for the three main sections of the reproducibility experiment. Additional textures are found in /AdditionalTextures. Spatial Tags are found in /SpatialTags.

# Requirements

**The following libraries are required:**

- matplotlib==3.2.1

- numpy==1.18.5

- scipy==1.4.1

- tensorflow-gpu==2.2.0

- Pillow==7.1.2

All my codes has been tested with Python 3.7 on Ubuntu 18.04.


\\
**Pretrained vgg-19 network**

A custom vgg network is used, as explained in the supplementals. It has been modified compared to the keras standard model:

- inputs are preprocessed (including normalization with imagenet stats).

- activations are scaled.

- max pooling layers are replaced with average pooling layers.

- zero padding is remplaced with reflect padding.

