---
layout: post
comments: true
title: "Module 2: Unsupervised appraoches for GANs interpretaion"
author: Manish Reddy Gottimukkula, Sonia Jaiswal
date: 2021-04-20
---


> Since the first GAN model[[1]](#reference) proposed by Ian GoodFellow, there has been a significant improvement in GANs to generate photorealistic images. However the domain of understanding and controlling the images being generated was not widely explored. Recently there are few different appraoches that have surfaced to interpret the image generation in deep generative models. Such approaches can be broadly categorized into supervised and unsupervised approaches. Supervised appraoches have come first and there was lot of analysis done on them. In contrast, this survey focuses on summarizing the very recent works done on unsupervised interpretation of GANs. We will discuss about identifying and editing the human intepretable concepts hidden in the image generation process of GANs. We will briefly discuss the unsupervised methods GANSpace[[2]](#reference), SeFa[[3]](#reference) and LatentCLR[[4]](#reference) to understand the recent works in this domain.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Generative Adversarial Networks have been widely succesful in generating varied images. Various GAN architectures like PGGAN[[6]](#reference), StyleGAN[[6]](#reference), BigGAN[[7]](#reference), StyleGAN2[[8]](#reference) have become popular for different kind of tasks. Most of these GANs generate images from random input latent vectors. Every GAN gains understanding of the kind of image to be generated based on the provided random input. So inherently, the input vectors holds the information needed by GAN to generate the corresponding image. It has been found [[9,10,11]](#reference) that when learning to generate images, the GANs represent multiple interpretable attributes in the latent space such as gender for face synthesis, lighting condition for scene synthesis etc. By understanding these interpretable dimensions we can modulate them later to control the image generation process. And also similar techniques can be extended to other deep generative models like VAE.

There exists both supervised and unsupervised interpretaion methods for GANs. Supervised methods focus on generating lots of images and hand labeling them to different categories and then train models to classify latent space. GANDissection[[12](#reference)] is one of the earliest such interpretion method where individual convolutional filters in the network mapped to the segmented objects in the output image are identified and then controlling the object placement by varying the individual units in the network. Later similar supervised interpretaion methods have been developed like InterfaceGAN[[13]](#reference) to edit facial attributes, Steerability of GAN attributes[[10]](#reference) etc. Major concern of supervised approaches involves in annotation of output images to various attribuets. To learn the latent representation of each attribute, the data has to be labeled and trained again. Even though we can learn very accurate latent semantics for various attributes, labeling overhead is critical with supervised approaches. In this paper, we won't go into the specifics on supervised approaches. Rather we analyse various unsupervised interpretation methods which doesn't require any labeling. <----write about GANSPace, SeFa, LatentCLR (one sentence about each)----> 


## GANSpace
## Semantic Factorization 
Semantic Factorization (SeFa) is a closed-form unsupervised method to discover interpretable dimensions in GANs. SeFa method doesn't involve any training and sampling of latent vectors like the GANSpace method, instead it takes a deeper look into the generation mechanism of GANs to understand the steerable directions in latent space. This method identifies the semantically meaningfully directions in the latent space by efficiently decomposing the model weights. Let G(.) be the generator function of the GAN which maps an input d-dimensional to an RGB image. G(.) consists of multiple layers and each learns a transformation from one to another. Let the first layer's output of GAN be G1 as shown below:

<img src="https://latex.codecogs.com/svg.image?\mathcal{Z}&space;\subseteq&space;\mathbb{R}^d&space;\to&space;\mathcal{I}&space;\subseteq&space;\mathbb{R}^{H&space;\times&space;W&space;\times&space;C}"/> \
<img src="https://latex.codecogs.com/svg.image?G_1(z)&space;=&space;y&space;=&space;Az&space;&plus;&space;b"/>

If we manipulate the input vector along the direction **n**, the output of first layer (y') will be updated as below:

<img src="https://latex.codecogs.com/svg.image?y'&space;=&space;G_1(z')&space;=&space;G_1(z&space;&plus;&space;\alpha&space;n)&space;=&space;Az&space;&plus;&space;n&space;&plus;&space;\alpha&space;An&space;=&space;y&space;&plus;&space;\alpha&space;An"/>

From above equation we can observe that manipulation process is instance independent, i.e given any latent vector z and a manipulation direction n, the manipulation effect can be obtained by adding the <img src="https://latex.codecogs.com/svg.image?\alpha&space;An"/> term to the output of the first layer. From this observation, this model confirms that A contains the essential knowledge of the image variation. As projection of direction vector **n** onto A results in variations of the image, finding the directions which results in large variations could be the most semantically meaningful directions learned by the GAN. Solving the below optimization problem will result in the top k meaningful directions:

<img src="https://latex.codecogs.com/svg.image?\large&space;N^{*}&space;=&space;argmax_{[N&space;\in&space;\mathbb{R}^{d&space;\times&space;k}:&space;\;&space;n_{i}^Tn_i&space;=&space;1&space;\;&space;\forall&space;i&space;=&space;1,...k]}&space;\;&space;\;&space;\sum_{i=1}^{k}\left\|&space;An_i&space;\right\|_2^2"/>

<img src="https://latex.codecogs.com/svg.image?\large&space;N^{*}&space;=&space;Eigenvectors(A^TA)"/>

Using above solution, this method shows a beautiful result which says that semantically meaningful directions can be obtained by finding the eigen vectors of the weight matrix A. Thus this method doesn't need both training and sampling. But we need human intervention to map the directions to corresponding human interpretable dimensions. As discussed before, annotation of discovered dimensions is the main challenge of unsupervised approaches. Also, the dimensions found could include entaglements of multiple interpretable elements and thus might not be useful for editing. Disentaglement of those dimensions is one of gaols of research in this direction.

![]({{ '/assets/images/team10/sefa1.png' | relative_url }})
Figure1: Semantic Factorization method results on StyleGAN and PGGAN.

## LatentCLR
## Comparision
## Conclusion
## Reference
[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Adv. Neural Inform. Process. Syst., 2014.

[2] Erik Harkonen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. In Adv. Neural Inform. Process. Syst., 2020.

[3] Yujun Shen, Bolei Zhou. Closed-Form Factorization of Latent Semantics in GANs. Computer Vision and Pattern Recognition (CVPR), 2021 

[4] Oğuz Kaan Yüksel, Enis Simsar, Ezgi Gülperi Er, Pinar Yanardag. LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions. ICCV 2021

[5] TeroKarras,TimoAila,SamuliLaine,andJaakkoLehtinen. Progressive growing of GANs for improved quality, stability, and variation. In Int. Conf. Learn. Represent., 2018.

[6] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., 2019.

[7] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019.

[8] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.

[9] LoreGoetschalckx,AlexAndonian,AudeOliva,andPhillip Isola. Ganalyze: Toward visual definitions of cognitive image properties. In Int. Conf. Comput. Vis., 2019.

[10] Ali Jahanian, Lucy Chai, and Phillip Isola. On the ”steerability” of generative adversarial networks. In Int. Conf. Learn. Represent., 2020.

[11] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Inter- preting the latent space of gans for semantic face editing. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.

[12] Bau, D., Zhu, J.Y., Strobelt, H., Zhou, B., Tenenbaum, J.B., Freeman, W.T., Torralba, A.: Gan dissection: Visualizing and understanding generative adversarial networks. International Conference on Learning Representations (2018)

[13] Shen, Y., Yang, C., Tang, X., Zhou, B.: Interfacegan: Interpreting the disentangled face representation learned by gans. IEEE Trans. on Pattern Analysis and Machine Intelligence (2020)

---
