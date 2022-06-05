---
layout: post
comments: true
title: "Module 4: AI robustness for Object Detection"
author: Weichong Ling, Yanxun Li, Zonglin Meng
date: 2021-04-27
---


>Object detection is an important vision task and has emerged as an indispensable component in many vision system, rendering its robustness as an increasingly important performance factor for practical applications. However, object detection models have been demonstrated to be vulnerable against various types of attack, it's significant to survey the current research attempts for a robust object detector model.
<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Object detection is an important computer vision task with plenty of real-world applications such as autonomous vehicles. While object detectors achieve higher and higher accuracy, their robustness has not been pushed to the limit. However, an object detector can be sensitive to natural noises such as sunlight reflections without demonstrating its robustness. Most significantly, it gives malicious parties a way to artificially fool the detector, thus leading to severe consequences. For example, [1] presents that sticking posters on the stop sign can make the object detector ignore it.

**[TODO] Other part of introduction**

To improve the robustness of a neural network, one common practice is adversarial training [2, 7]. It achieves robust model training by solving a minimax problem, where the inner maximization generates attacks according to the current model parameters while the outer optimization minimizes the training loss with respect to the model parameters. Zhang et al. [8] extend adversarial training to the object detection domain by leveraging attack sources from both classification and localization. Chen et al. [6] decompose the total adversarial loss into class-wise losses and normalize each class loss using the number of objects for the class. Det-AdvProp [4] achieves object detection robustness with a different approach. It improves the model-dependent data augmentation [5] and fits it into the object detection domain. Different from the previous work, Det-AdvProp considers the common practice of pre-train and fine-tine two-step paradigm. It performs data augmentation during the fine-tuning stage without touching the resource-consuming pre-train stage.
				
This survey aims at selecting and summarizing object detection research from two perspectives. First, we briefly review the typical structure and the learning objective of object detection. Next, we are going to look into some attack attempts in object detection, compare their results, and discuss the potential threats to real-world scenarios. Then, we dive into the domain of robust training of object detectors. 

## Adversarial Training for Object Detection


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

[1] Song, Dawn, et al. "Physical adversarial examples for object detectors." 12th USENIX workshop on offensive technologies (WOOT 18). 2018.<br>
[2] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).<br>
[3] Haichao Zhang and Jianyu Wang. Towards adversarially robust object detection. In International Conference on Computer Vision, 2019 <br>
[4] Chen, Xiangning, et al. "Robust and accurate object detection via adversarial learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.	<br>
[5] Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan L. Yuille, and Quoc V. Le. Adversarial examples improve image recognition. In Computer Vision and Pattern Recognition, 2020. <br>
[6] Pin-Chun Chen, Bo-Han Kung, Jun-Cheng Chen; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 10420-10429<br>		
[7] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.<br>
[8] Zhang, Haichao, and Jianyu Wang. "Towards adversarially robust object detection." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.<br>

---

