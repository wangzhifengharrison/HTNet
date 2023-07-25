# HTNet for micro-expression recognition

A  Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement.

Example of micro-expressions:

<img src="https://github.com/christy1206/biwoof/blob/pictures/006_006_1_2.gif" width="200" height="200"/> <img src="https://github.com/christy1206/biwoof/blob/pictures/s03_s03_po_11.gif" width="200" height="200"/> <img src="https://github.com/christy1206/biwoof/blob/pictures/sub11_EP15_04f.gif" width="200" height="200"/>

SAMM (006_006_1_2), SMIC (s03_s03_po_11), CASME II (sub11_EP15_04f)

STSTNet is a two-layer neural network that is capable to learn the features from three optical flow features (horizontal optical flow images, vertical optical flow images and optical strain) computed from the onset and apex frames from each video. Please find the source code for optical flow adopted in this experiment at http://www.ipol.im/pub/art/2013/26/

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
