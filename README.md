# HTNet for micro-expression recognition

A  Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement.

Facial expression is related to facial muscle contractions and different muscle movements correspond to different emotional states.  For micro-expression recognition, the muscle movements are usually subtle, which has a negative impact on the performance of current facial emotion recognition algorithms.  Most existing methods use self-attention mechanisms  to capture relationships between tokens in a sequence, but they do not take into account the inherent spatial relationships between facial landmarks. This can result in sub-optimal performance on  micro-expression recognition tasks.Therefore, learning to recognize facial muscle movements is a key challenge in the area of micro-expression recognition.  In this paper, we propose a Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement.  HTNet includes two major components: a transformer layer that leverages the local temporal features and an aggregation layer that extracts local and global semantical facial features.  Specifically, HTNet divides the face into four different facial areas: left lip area, left eye area, right eye area and right lip area.  The transformer layer is used to focus on representing local minor muscle movement with local self-attention in each area.  The aggregation layer is used to learn the interactions between eye areas and lip areas. The experiments on four publicly available micro-expression datasets show that the proposed approach outperforms previous methods by a large margin.

<img src="https://github.com/christy1206/STSTNet/blob/picture/flow.JPG" width="500" height="400"/>

The recognition results achieved are:

<img src="https://github.com/christy1206/STSTNet/blob/picture/result.JPG" width="600" height="150"/>

The databases include CASME II (145 videos), SMIC (164 videos) and SAMM (133 videos). "Full" is the composite database of the 3 databases (442 videos).


The exact configuration of STSTNet is:

<img src="https://github.com/christy1206/STSTNet/blob/picture/configuration.JPG" width="500" height="200"/>

## Python code

Pytorch framework is used to reproduce the result. Note that the results obtained are slightly lower from the original work. The result for Full (Composite) is UF1: 0.7209 and UAR: 0.725.

<b>Step 1)</b> Please download the dataset from https://bit.ly/2S35u05 and put it in /datasets

<b>Step 2)</b> Place the files in the structure as follows:
>├─datasets <br>
>|--three_norm_u_v_os
>|--combined_datasets_whole
>├─main_HTNet.py <br>
>├─requirements.txt <br>

<b>Step 3)</b> Installation of packages using pip

``` pip install -r requirements.txt ```

<b>Step 4)</b> Training and Evaluation

``` python main_HTNet.py --train False```


Thank you for your interest and support.
