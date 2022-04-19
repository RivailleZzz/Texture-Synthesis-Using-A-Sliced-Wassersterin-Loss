# Texture Synthesis Using A Sliced Wassersterin Loss

![image](https://user-images.githubusercontent.com/97613092/164106212-ec6816ad-dacd-48ac-beae-1353d6a56f1f.png)

<Br/>

In this document, I will briefly introduce the requirements, models and parameters and the results of my experiments. The results of synthesized textures using traditional machine learning methods are also presented here. More detailed results and the corresponding explanations are available in my paper.

<Br/>

The original files are very limited, I've collected all the provided codes and files to the folder "**1. original codes and files**", and I summarized all my codes (implementation, innovations and traditional machine learning method) in the folder "**2. Texture Synthesis Using A Sliced Wassersterin Loss**". The names of all the codes in the folder correspond to their functions, so that we can conveniently find and study them.

<Br/>

To get and study the relevant results of these texture related tasks, run the Colab notebooks:

- style Transfer.ipynb

- Texture Synthesis.ipynb

- Texture with tags.ipynb

For the three main sections of the reproducibility experiments, the images used in my experiments and some additional textures can be found in /AdditionalTextures and /Styletransfer. Spatial Tags can be found in /SpatialTags.

<Br/>

For the traditional machine learning method, run the Colab notebook:

- Traditional Machine Learning Methods.ipynb

<Br/>


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

<Br/>

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

<Br/>

# Results


**Texture Synthesis:**

![image](https://user-images.githubusercontent.com/97613092/164103989-e34b056c-40f6-413d-8a9b-93469e04a445.png)

<Br/>

**Style Transfer:**

![image](https://user-images.githubusercontent.com/97613092/164104446-f418776b-92ac-4491-b125-7381ae0dafd1.png)

<Br/>

**Spatial Constraints with Tags**

![image](https://user-images.githubusercontent.com/97613092/164105655-43f3527b-319c-4d4b-8442-383396ed8e05.png)

<Br/>

**Loss Function Convergence and Time-complexity**

![image](https://user-images.githubusercontent.com/97613092/164105523-39e6d9d6-1d87-4cef-8b59-a401cd77a955.png)

<Br/>

# Traditional Machine Learning Method

**Results using KNN:**

![image](https://user-images.githubusercontent.com/97613092/164106060-7d2f9c7b-c0ac-497a-a849-f02a5d60771f.png)
