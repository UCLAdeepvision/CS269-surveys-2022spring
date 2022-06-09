---
layout: post
comments: true
title: "Module 2: Human-AI Creation - Text to Image Generation"
author: Vishnu Vardhan Bachupally, Sri Keerthi Bolli
date: 2022-06-09
---


> Text to image generation has been famous in recent times
with the advancements in the generative models like GANs,
Autoregressive models, Diffusion based models, etc. There
have been exciting papers on text to image generation using
these concepts. One such paper is Imagen (7) that has been
recently released by Google research. In this work we aim
to provide a survey of methods used to achieve the text to
image generation task comparing them both qualitatively
and quantitatively.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
There have many works that tried to achieve the task of text to image generation. Few works like Imagen( 7), DALLE( 5) have tried to obtain the images from plain text. Few works tried to use additional information such as segmentation maps like Make-a-scene (2 ). The research of the image generation started with RNN based models where the pixels were generated sequentially and attention was also used to create better images like in DRAW (3 ). There have been multiple GAN based approaches for the text to image generation task. Some examples include DM-GAN ( 10 ) that introduced a dynamic memory component, DF-GAN (8) that fused text information into image features, XMC-GAN (9 ) that used contrastive learning to maximize the mutual information between image and text. There were few au- toregressive models that were devised to perform this task like DALL-E (6), CogView (1) and Make-a-scene (2 ) that train on text and image tokens. Finally there are diffusion based models that are very good at generating photorealistic images. Diffusion based models are being used in state of the art methods like GLIDE (4), DALL-E 2 ( 5 ) and Imagen. We explain a few models in each of the above different methods using GANs, Autoregressive models, and Diffu- sion based models. We also provide qualitative comparisons among the various models mentioned above to give the reader a better understanding of the improvements in the latest state-of-the art methods compared to previous methods.

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
