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

Option 3: Download BowFire's dataset from given link: [Click here](https://drive.google.com/file/d/1g1db4Z6OQTBlngnxXVpF2rd22vHGZwc7/view?usp=drive_link) (Recommended to use batchsize of 8 during training)

Option 4: The proposed DFAN dataset [Click here](https://drive.google.com/file/d/10z998vuTzkNJElZdsSDrbIDpRWJ4aZoo/view?usp=drive_link)

### 4. Qualitative Results

### 5. Quantitative Results
*Notice:* Please follow the paper pdf to view the references.


## 5. Citation and Acknowledgements
Please read and cite our following papers on Fire Detection if you like our work:

<pre>
<code>
@article{yar2022optimized,
  title={Optimized dual fire attention network and medium-scale fire classification benchmark},
  author={Yar, Hikmat and Hussain, Tanveer and Agarwal, Mohit and Khan, Zulfiqar Ahmad and Gupta, Suneet Kumar and Baik, Sung Wook},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={6331--6343},
  year={2022},
  publisher={IEEE}
}
}</code>
</pre>

<pre>
<code>
@article{yar2023modified,
  title={A modified YOLOv5 architecture for efficient fire detection in smart cities},
  author={Yar, Hikmat and Khan, Zulfiqar Ahmad and Ullah, Fath U Min and Ullah, Waseem and Baik, Sung Wook},
  journal={Expert Systems with Applications},
  volume={231},
  pages={120465},
  year={2023},
  publisher={Elsevier}
}
}</code>
</pre>

<pre>
<code>
  @article{yar2023effective,
  title={An Effective Attention-based CNN Model for Fire Detection in Adverse Weather Conditions},
  author={Yar, Hikmat and Ullah, Waseem and Khan, Zulfiqar Ahmad and Baik, Sung Wook},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={206},
  pages={335--346},
  year={2023},
  publisher={Elsevier}
}
}</code>
</pre>


## 6. Contact
I would be happy to guide and assist in case of any questions and I am open to research discussions and collaboration in Fire Detection domain. Ping me at tanveer445 [at] [ieee] [.org]

