# Texture Synthesis Using A Sliced Wassersterin Loss

This is the official implementation of "A Sliced Wasserstein Loss for Neural Texture Synthesis" paper (to appear in CVPR 2021).

![image](https://user-images.githubusercontent.com/97613092/164089466-5b4b8796-419c-44fa-a825-bb02d046859b.png)

Run the Colab notebooks:

- style Transfer.ipynb

- Texture Synthesis.ipynb

- Texture with tags.ipynb

For the three main sections of the reproducibility experiments, the images used in my experiments and some additional textures can be found in /AdditionalTextures and /Styletransfer. Spatial Tags are found in /SpatialTags.

<Br/>

For the traditional machine learning method, run the Colab notebook:

- Traditional Machine Learning Methods.ipynb

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

# Texture related parameters

**The imput parameters are:**

- iters: number of calls to l-bfgs (by default: 20). Each step is one call to scipy's l-bfgs implementation with maxfun=64.
- size: the input texture will be resized to this size (by default: 256, which resizes to 256x256). If the image is not square, it will be center-cropped. The - generated texture will have the same resolution.
- output: name of the output file (by default: output.jpg).
- filename: name of the input texture (only mandatory parameter).

<Br/>

**The output parameters are:**

- resized-input.jpg: input image is resized following the --size parameter. It will be exported as "resized-input.jpg" so it can be compared with the generated output.
- output file: final output file after all iterations. The name is specified by the --output tag (output.jpg by default).
- output-iterN.jpg: the intermediate result after N iterations. If there are 20 iterations, there will be 20 output images.

# Results:

**Texture Synthesis:**

![image](https://user-images.githubusercontent.com/97613092/164103989-e34b056c-40f6-413d-8a9b-93469e04a445.png)

<Br/>

**Style Transfer:**

![image](https://user-images.githubusercontent.com/97613092/164104446-f418776b-92ac-4491-b125-7381ae0dafd1.png)

<Br/>

**Spatial Constraints with Tags**

![image](https://user-images.githubusercontent.com/97613092/164105031-3f0fedb0-d906-4e1c-b7c0-c0dd9c361dca.png)

<Br/>

**Loss Function Convergence and Time-complexity**

![image](https://user-images.githubusercontent.com/97613092/164105523-39e6d9d6-1d87-4cef-8b59-a401cd77a955.png)
