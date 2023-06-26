# DFAN
Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark



# DFAN (TIP2022)
## Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark

This paper has been accepted to IEEE Transactions on Image Processing

To download the framework: [DFAN.pdf]()

### 1. Paper Links
https://ieeexplore.ieee.org/abstract/document/9898909

IEEE-TIP: 
## 2. Setup
You need to install Tensorflow (preferred 2.9.0) and some basic libraries including PIL, cv2, numpy, etc.
For installation used 
pip install -r requirements.txt

## 3. How to Train?
In the repository, the DFAN.ipynb file is used to train the orignal DFAN model, where the Attention_mechanism_with_InceptionV3_github_compress_de is used for compress version. 

### 3.1. Datasets
The datasets can be downloaded from the following links. We follow the training and testing data similar to the previous methods.

Option 1: Download FD dataset from given link: [Click here](http://www.nnmtl.cn/EFDNet/)

Option 2: Download Foggia's dataset from given link: [Click here](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)

Option 3: Download BowFire's dataset from given link: [Click here](https://bitbucket.org/gbdi/bowfire-dataset/downloads/)

### 3.2. Training and Testing
Run train.py ()

### 4. Qualitative Results

### 5. Quantitative Results
*Notice:* Please follow the paper pdf to view the references.


## 5. Citation and Acknowledgements
Please read and cite our following papers on Fire Detection if you like our work:

<pre>
<code>
@article{yar2021vision,
  title={Vision sensor-based real-time fire detection in resource-constrained IoT environments},
  author={Yar, Hikmat and Hussain, Tanveer and Khan, Zulfiqar Ahmad and Koundal, Deepika and Lee, Mi Young and Baik, Sung Wook},
  journal={Computational intelligence and neuroscience},
  volume={2021},
  year={2021},
  publisher={Hindawi}
}
}</code>
</pre>

<pre>
<code>@article{yar2021fire,
  title={Fire Detection via Effective Vision Transformers},
  author={Yar, Hikmat and Hussain, Tanveer and Khan, Zulfiqar Ahmad and Lee, Mi Young and Baik, Sung Wook},
  journal={The Journal of Korean Institute of Next Generation Computing},
  volume={17},
  number={5},
  pages={21--30},
  year={2021}
}</code>
</pre>


## 6. Contact
I would be happy to guide and assist in case of any questions and I am open to research discussions and collaboration in Fire Detection domain. Ping me at tanveer445 [at] [ieee] [.org]

