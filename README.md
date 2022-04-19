# Texture Synthesis Using A Sliced Wassersterin Loss

This is the official implementation of "A Sliced Wasserstein Loss for Neural Texture Synthesis" paper (to appear in CVPR 2021).

![image](https://user-images.githubusercontent.com/97613092/164089466-5b4b8796-419c-44fa-a825-bb02d046859b.png)

Run the Colab notebooks:

- style_transfer.ipynb

- texture_synthesis.ipynb

- texture_synthesis_with_tags.ipynb

For the three main sections of the reproducibility experiment. Additional textures are found in /AdditionalTextures. Spatial Tags are found in SpatialTags.

# Requirements

**The following libraries are required:**

- matplotlib==3.2.1

- numpy==1.18.5

- scipy==1.4.1

- tensorflow-gpu==2.2.0

- Pillow==7.1.2

All my codes has been tested with Python 3.7 on Ubuntu 18.04.  

<Br/>

**Pretrained vgg-19 network**

A custom vgg network is used, as explained in the supplementals. It has been modified compared to the keras standard model:

- inputs are preprocessed (including normalization with imagenet stats).

- activations are scaled.

- max pooling layers are replaced with average pooling layers.

- zero padding is remplaced with reflect padding.

# Texture synthesis 

**The parameters are:**

- iters: number of calls to l-bfgs (by default: 20). Each step is one call to scipy's l-bfgs implementation with maxfun=64.
- size: the input texture will be resized to this size (by default: 256, which resizes to 256x256). If the image is not square, it will be center-cropped. The - generated texture will have the same resolution.
- output: name of the output file (by default: output.jpg).
- filename: name of the input texture (only mandatory parameter).

<Br/>

**Outputs files are:**

- resized-input.jpg: input image is resized following the --size parameter. It will be exported as "resized-input.jpg" so it can be compared with the generated output.
- output file: final output file after all iterations. The name is specified by the --output tag (output.jpg by default).
- output-iterN.jpg: the intermediate result after N iterations. If there are 20 iterations, there will be 20 output images.
