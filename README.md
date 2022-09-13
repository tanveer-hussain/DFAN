# DFAN
Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark



# DFAN (TIP2022)
## Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark

This paper has been accepted to IEEE Transactions on Image Processing

To download the framework: [DFAN.pdf]()

### 1. Paper Links
ArXiv: https://arxiv.org/abs/2204.06788

CVPRW: https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Hussain_Pyramidal_Attention_for_Saliency_Detection_CVPRW_2022_paper.pdf

## 2. Setup
You need to install Tensorflow (preferred 2.9.0) and some basic libraries including PIL, cv2, numpy, etc.

## 3. How to Train?

### 3.1. Datasets
The datasets can be downloaded from the following links. We follow the training and testing data similar to the previous methods.

Option 1: Download FD dataset from given link: [Click here](http://www.nnmtl.cn/EFDNet/)

Option 2: Download Foggia's dataset from given link: [Click here](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)

Option 3: Download BowFire's dataset from given link: [Click here](https://bitbucket.org/gbdi/bowfire-dataset/downloads/)

### 3.2. Training and Testing
Run train.py (current training code supports RGB-D dataset training, you can request for RGB as well or tune the code yourself)

In the training code, there is function call to the testing code which then evaluates the model's performance and stores the results in corresponding folders.

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
<code>@misc{hussain2021densely,
      title={Densely Deformable Efficient Salient Object Detection Network}, 
      author={Tanveer Hussain and Saeed Anwar and Amin Ullah and Khan Muhammad and Sung Wook Baik},
      year={2021},
      eprint={2102.06407},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</code>
</pre>

Some of the functions in codes are inspired by [UCNet](https://github.com/JingZhang617/UCNet) GitHub repository. The authors are thankful for their nice and explained SOD GitHub page.

## 6. Contact
I would be happy to guide and assist in case of any questions and I am open to research discussions and collaboration in Saliency Detection domain. Ping me at tanveer445 [at] [ieee] [.org]

