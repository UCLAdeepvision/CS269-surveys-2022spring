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

There are many ways of adversarially attacking an object detector. One common way is adding adversarial perturbations as visible stickers or patches to the objects either virtually or physically. Such perturbation can successfully fool detectors, but are perceptible to humans. Another way of adversarial attack is adding human-imperceptible perturbations to the image, which can result in detectors failing to detect objects or detect wrong objects.

To improve the robustness of a neural network, one common practice is adversarial training [2, 7]. It achieves robust model training by solving a minimax problem, where the inner maximization generates attacks according to the current model parameters while the outer optimization minimizes the training loss with respect to the model parameters. Zhang et al. [8] extend adversarial training to the object detection domain by leveraging attack sources from both classification and localization. Chen et al. [6] decompose the total adversarial loss into class-wise losses and normalize each class loss using the number of objects for the class. Det-AdvProp [4] achieves object detection robustness with a different approach. It improves the model-dependent data augmentation [5] and fits it into the object detection domain. Different from the previous work, Det-AdvProp considers the common practice of pre-train and fine-tine two-step paradigm. It performs data augmentation during the fine-tuning stage without touching the resource-consuming pre-train stage.
				
This survey aims at selecting and summarizing object detection research from two perspectives. First, we briefly review the typical structure and the learning objective of object detection. Next, we are going to look into some attack attempts in object detection, compare their results, and discuss the potential threats to real-world scenarios. Then, we dive into the domain of robust training of object detectors.

## Adversarial Attacks for Object Detection

The common practice of generating adversarial perturbations against object detectors is to optimize detection or classification loss. [1] extended PR2 algorithm and utilized it to generate physical adversarial posters and stickers (Fig.1) to attack YOLO v2 object detector. Physical objects with the posters and stickers attached to them will be misclassified. [1] also generated physical adversarial patches which can fool YOLO v2 into detecting non-existent objects. 

![stop]({{ '/assets/images/Module04OD/stop.png' | relative_url }})
{: style="width: 300px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.1 Left two images: Output of the extended RP2 algorithm to attack YOLO v2 using poster and sticker attacks. Right image: Patch aimed at fooling YOLO v2 into detecting nonexistent Stop signs.</i></center>
<br>

[9] extended DPatch method into Clipped Attack method to generate adversarial patches against YOLO v3. The adversarial patches can be either attached to an image or placed in the physical world. In both cases, the adversarial patches successfully confused the object detector into focusing on the patches and ignoring detectable objects (Fig.2). Quantitatively, the Clipped Attack method caused YOLO v3 to drop from 55.4 to single digit mAP (mean average precision).

![virtual_attack]({{ '/assets/images/Module04OD/virtual_attack.png' | relative_url }})
{: style="width: 300px; max-width: 100%; margin-left: auto; margin-right: auto"}
![physical_attack]({{ '/assets/images/Module04OD/physical_attack.png' | relative_url }})
{: style="width: 300px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.2 Top two rows: ROI plots for the clipped case. Bottom row: Physical attack using the patch.</i></center>
<br>

[10] proposed DAG algorithm (Fig.3) to effectively generate visually imperceptible perturbations, and confused the originally correct recognition results in a well controllable manner (Fig.4). An important property of DAG is that it can generate transferable adversarial perturbations. The perturbations can be transferred across different training sets, different network architectures and even different tasks. Consequently, it has caused several object detection networks to drop to single digit mAP. The performance and transferability of DAG has proven the effectiveness of adversarial attack.

![DAG]({{ '/assets/images/Module04OD/DAG.png' | relative_url }})
{: style="width: 400px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.3 DAG Algorithm.</i></center>
<br>

![DAG_output]({{ '/assets/images/Module04OD/DAG_output.png' | relative_url }})
{: style="width: 300px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.4 An adversarial example for semantic segmentation and object detection. FCN is used for segmentation, and Faster-RCNN is used for detection. Left column: the original image (top row) with the normal segmentation (the purple region is predicted as dog) and detection results. Right column: after the adversarial perturbation (top row, magnified by 10) is added to the original image, both segmentation (the light green region as train and the pink region as person) and detection results are completely wrong. Note that, though the added perturbation can confuse both networks, it is visually imperceptible (the maximal absolute intensity in each channel is less than 10).</i></center>
<br>


## Adversarial Training for Object Detection

### Improve Robustness by adversarial training

#### The role of task losses in adversarial training

Although robustness can be improved by adversarial training [2,7], it’s much tricker in object detection. Different from adversarial training on classification tasks which usually has only one loss, object detection tasks consist of two losses (Fig.4). One is the classification loss and the other is localization loss. It’s natural to ask, should we perform the adversarial attack on the sum of them or each of them? Both [3] and [4] experiment with this problem. 

![loss1]({{ '/assets/images/Module04OD/loss.png' | relative_url }})
{: style="width: 400px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.4 Loss function of object detector training</i></center>
<br>

[3] attacks the classification loss and the localization loss respectively. The result indicates that two attacks have mutual impacts and the adversarial attacks trailered for one task can reduce the performance of the model on the other task. For example, the attack on the classification not only hurts classification accuracy but also drops localization IoU significantly (Fig.5a). This observation has two indications. First, attacking an object detector is easy. Even attacks on one individual task loss can effectively impede the performance as a whole. Second, we have to study the interaction of these two losses. When generating adversarial examples by attacking both losses, whether the gradients of two losses counteract or enhance will significantly impact the quality of adversarial examples.

![lossdrop]({{ '/assets/images/Module04OD/lossdrop.png' | relative_url }})
{: style="width: 500px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.5 Marginal performance drop in attacking two tasks loss individually</i></center>
<br>

[3] conducts the second experiment to answer this question. In Fig.5b, we can draw two conclusions. (1) the magnitudes of the task gradients are not the same. They spread onto different value ranges, indicating the potential existence of imbalance between two losses; (2) the directions of two gradients are inconsistent (non-diagonal), implying the potential conflicts between them. [4] also does a similar experiment. In Fig.6, LOC, CLS, and DET denote attacks on localization, classification, and both respectively. We can see that the highest mAP is not achieved by the row DET, indicating two losses cancel out each other.

![tab1]({{ '/assets/images/Module04OD/tab1.png' | relative_url }})
{: style="width: 400px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.6 mAP score for attacking LOC, CLS, DET</i></center>
<br>

The solution to this is simple. Both [3] and [4] proposed a similar algorithm regarding this issue. Here we present the algorithm in [3]. Intuitively, it attacks localization and classification loss respectively, and only keeps the one that maximizes the total loss. The adversarial output then replaces the original sample in a normal training process. 

![Algo1]({{ '/assets/images/Module04OD/Algo1.png' | relative_url }})
{: style="width: 400px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.7 Algorithm of adversarial training for robust object detectors [3] </i></center>
<br>

#### To improve the algorithm

Based on the algorithm proposed in [3], [4] has two improvements. First, it puts robust object detector training in the context of the pretrain-finetune paradigm. Second, it argues that the adversarial samples have different distributions from the original images. The batch norm layers have to take this discrepancy into account.

Using adversarial examples to augment data can be taken place in two periods. One is during the pretraining, and the other is during finetuning. [4] argues that although adversarial examples boost up robustness on pretraining, such performance gains cannot transfer after finetuning. Fig.8 shows that pretraining on AdvProp[5] and Noisy Student[11] have similar or even worse performance after finetuning (D2 and D5 are two object detection models where D2 is more light-weighted). Therefore, [4] purposed to move the adversarial data augmentation step from pretraining to finetuning. Fig.9 shows that finetuning 6 models with Det-AdvProp surpasses the vanilla baseline.

![bar]({{ '/assets/images/Module04OD/bar.png' | relative_url }})
{: style="width: 400px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.8  mAP comparsion for vanilla model and finetuned pretrained models. The data augmentation is at pretraining step</i></center>

![tab2]({{ '/assets/images/Module04OD/tab2.png' | relative_url }})
{: style="width: 500px; max-width: 100%; margin-left: auto; margin-right: auto"}
<center><i>Fig.9  mAP comparsion for vanilla model and Det-AdvProp finetuned models. The data augmentation is at finetuning step</i></center>
<br>

Another improvement to [3] is that [4] draws adversarial examples from another distribution. [4] computes the distribution of adversarial examples in an online manner. Note that adversarial examples for classification loss and classification loss are drawn from the same distribution. Therefore, [4] manages two distributions. One is for the original images and the other is for adversarial examples. One last tiny difference is that [4] does not replace the original image with an adversarial example. Instead, it puts both samples into the normal training process.

In conclusion, both papers demonstrate that traditional adversarial training of classification tasks can be transferred to train a robust object detector. When generating adversarial examples, it’s suggested to attack classification or localization separately and pick the one that maximizes the total loss. In the context of pretraining and finetuning, [4] indicates that augmenting data during finetune step can better preserve the performance gain during pretraining.


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
[9] Mark Lee and Zico Kolter. "On Physical Adversarial Patches for Object Detection." Computer Vision and Pattern Recognition. 2019.<br>
[10] Cihang Xie, Jianyu Wang, Zhishuai Zhang, Yuyin Zhou, Lingxi Xie, Alan Yuille; "Adversarial Examples for Semantic Segmentation and Object Detection." Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 1369-1378. <br>

---

