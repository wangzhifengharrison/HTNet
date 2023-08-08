# HTNet for micro-expression recognition
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htnet-for-micro-expression-recognition/micro-expression-recognition-on-casme3)](https://paperswithcode.com/sota/micro-expression-recognition-on-casme3?p=htnet-for-micro-expression-recognition)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htnet-for-micro-expression-recognition/micro-expression-recognition-on-casme-ii-1)](https://paperswithcode.com/sota/micro-expression-recognition-on-casme-ii-1?p=htnet-for-micro-expression-recognition)

Official implementation of our paper:  
**HTNet for micro-expression recognition**  
Zhifeng Wang, Kaihao Zhang, Wenhan Luo, Ramesh Sankaranarayana 
[[paper]](https://arxiv.org/abs/2307.14637)

A  Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement.

Facial expression is related to facial muscle contractions and different muscle movements correspond to different emotional states.  For micro-expression recognition, the muscle movements are usually subtle, which has a negative impact on the performance of current facial emotion recognition algorithms.  Most existing methods use self-attention mechanisms  to capture relationships between tokens in a sequence, but they do not take into account the inherent spatial relationships between facial landmarks. This can result in sub-optimal performance on  micro-expression recognition tasks.Therefore, learning to recognize facial muscle movements is a key challenge in the area of micro-expression recognition.  In this paper, we propose a Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement.  HTNet includes two major components: a transformer layer that leverages the local temporal features and an aggregation layer that extracts local and global semantical facial features.  Specifically, HTNet divides the face into four different facial areas: left lip area, left eye area, right eye area and right lip area.  The transformer layer is used to focus on representing local minor muscle movement with local self-attention in each area.  The aggregation layer is used to learn the interactions between eye areas and lip areas. The experiments on four publicly available micro-expression datasets show that the proposed approach outperforms previous methods by a large margin.

<p align="center">
  <img src="https://github.com/wangzhifengharrison/HTNet/blob/master/images/micro-architecture.png" width="700" height="480"/>
</p>
HTNet: Overall architectures of hierarchical transformer network for micro-expression recognition.Low-level self-attention in transformer layerscaptures fine-grained features in local regions. High-level self-attention in transformer layers captures coarse-grained features in global regions. An aggregation block isproposed to create interactions between different blocks at the same level.

The experiments are implemented on SAMM[32], SMIC[33], CASME II[34] and CASME III [35] databases. SAMM, SMIC, and CASME II are merged into one composite dataset,and the same labels in these three datasets are adopted for micro-expression tasks. In these datasets, the “positive” emotion category includes the “happiness” emotion class, and the “negative” emotion category includes “sadness”,“disgust”, “contempt”, “fear” and “anger”
emotion classes while “surprise” emotion category only includes “surprise” class:
<p align="center">
<img src="https://github.com/wangzhifengharrison/HTNet/blob/master/images/datasets.png" width="500" />
</p>

The Unweighted F1-score (UF1) and Unweighted Average Recall (UAR) performance of handcraft methods, deep learning methods and our HTNet method under LOSO protocol on the composite (Full), SMIC, CASME II and SAMM. Bold text indicates the best result.

The results are listed as follows:
<p align="center">
  <img src="https://github.com/wangzhifengharrison/HTNet/blob/master/images/state_of_art.png" width="500" />
</p>


We investigate the effects of the transformer layer’s head count on accuracy in composite datasetsSMIC, SAMM and CASME II. The composite datasets’ Unweighted F1-score (UF1) and Unweighted Average Recall (UAR) performance are reported.
<p align="center">
<img src="https://github.com/wangzhifengharrison/HTNet/blob/master/images/effects_trasformer.png" width="500"/>
</p>

## Python code


<b>Step 1)</b> Please download the dataset and put it in /datasets

<b>Step 2)</b> Place the files in the structure as follows:
>├─datasets <br>
>--three_norm_u_v_os <br>
>--combined_datasets_whole <br>
>├─main_HTNet.py <br>
>├─requirements.txt <br>

<b>Step 3)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 4)</b> Training and Evaluation

``` python main_HTNet.py --train True```

# Citation
If you find our work useful for your project, please consider citing the paper<br>
```bibtex
@misc{wang2023htnet,
      title={HTNet for micro-expression recognition}, 
      author={Zhifeng Wang and Kaihao Zhang and Wenhan Luo and Ramesh Sankaranarayana},
      year={2023},
      eprint={2307.14637},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Thank you for your interest and support.
