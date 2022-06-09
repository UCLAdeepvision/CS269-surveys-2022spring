---
layout: post
comments: true
title: "Module 10: Debate on Explainable ML"
author: Andong Hua, Boya Ouyang, Jiayue Sun, and Yu-Hsuan Liu 
date: 2022-06-08
---


> Nowadays, deep neural networks are widely used to build machine learning models and AI, and their applications are common in daily life including chatbox, object detection, etc. People want to interpret the backbox of the model to understand what those models learn and see. However, people tend to overexplain the association between the result and the model or over-rely on the interpretation method such as model properties or post-hoc interpretation techniques. In this survey, we focus on analyzing several feature attribution based interpretation methods. We would like to discuss how people evaluate those methods and how those methods might mislead people.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
With the rapid development of machine learning and AI, more and more applications adopt ML and AI. For example, object detection, image classification, image generations, etc. However, the models are the black box to the human. As the dimension and the number of parameters grow insanely to increase the performance, it is hard for a human to realize what the models learn and what the model analyzes and processes. Therefore, people start trying to interpret the model. For example, they try to build the association between the results and the model properties. Post-hoc interpretation techniques are invented to help with the explanation of how the model works. With a better understanding of the models, people believe that they can adjust the model based on the demand. However, the end goal of the interpretation models has no formal definition. Yet we do not know whether the interpretation models actually improve the model’s performance, robustness, or generalization.

In the next section, we go over the issues needed to be solved with the definition and the desiderata of interpretability and a brief introduction to the current interpretation techniques. Since many interpretation techniques are proposed to explain the models, we focus on the analysis of feature attribution based methods, which is one example of post-hoc interpretation, in the following section. 

## Background


## Feature Attribution Based Methods


## Conclusion



## Reference
[1] Zachary C. Lipton. ["The Mythos of Model Interpretability."](http://arxiv.org/abs/1606.03490) 2016.

[2] Kim, Been. ["Interactive and interpretable machine learning models for human machine collaboration."](https://dspace.mit.edu/handle/1721.1/98680) *PhD thesis, Massachusetts Institute of Technology*. 2015. 

[3] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. [""Why Should I Trust You?": Explaining the Predictions of Any Classifier."](http://arxiv.org/abs/1602.04938) 2016.

[4] Liu, Changchun, Rani, Pramila, and Sarkar, Nilanjan. ["An empirical study of machine learning techniques for affect recognition in human-robot interaction."](https://ieeexplore.ieee.org/document/1545344) *In International Conference on Intelligent Robots and Systems*. IEEE, 2005. 

[5] Bryce Goodman, and Seth Flaxman. ["European Union regulations on algorithmic decision-making and a "right to explanation"."](http://arxiv.org/abs/1606.08813) 2016, AI Magazine, Vol 38, No 3, 2017.

[6] S. Krening, B. Harrison, K. M. Feigh, C. L. Isbell, M. Riedl and A. Thomaz. "Learning From Explanations Using Sentiment and Advice in RL." *In IEEE Transactions on Cognitive and Developmental Systems*, vol. 9, no. 1, pp. 44-55, March 2017.

[7] Wojciech Samek, Alexander Binder, Grégoire Montavon, Sebastian Bach, and Klaus-Robert Müller. ["Evaluating the visualization of what a Deep Neural Network has learned."](http://arxiv.org/abs/1509.06321) 2015.


[8] K. Simonyan, A. Vedaldi, and A. Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." *In Proc. ICLR Workshop*, 2014, pp. 1–8.

[9] M. D. Zeiler and R. Fergus. "Visualizing and understanding convolutional networks." *In Proc. ECCV*, 2014, pp. 818–833.

[10] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek. "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation." *PLOS ONE*, vol. 10, no. 7, p. e0130140, 2015.

[11] Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. ["Sanity Checks for Saliency Maps."](http://arxiv.org/abs/1810.03292) 2018.

[12] Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim. ["A Benchmark for Interpretability Methods in Deep Neural Networks."](http://arxiv.org/abs/1806.10758) 2018.

[13] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. ["SmoothGrad: removing noise by adding noise."](http://arxiv.org/abs/1706.03825) 2017.

[14] Adebayo, Julius, Michael Muelly, Harold Abelson, and Been Kim. ["Post hoc explanations may be ineffective for detecting unknown spurious correlation."](https://openreview.net/forum?id=xNOVfCCvDpM) *In International Conference on Learning Representations*. 2021.

---
