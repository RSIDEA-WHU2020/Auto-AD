 <h1 align="center"> Project Auto-AD </h1>
This is an official implementation of Auto-AD in our TGRS 2021 paper " Auto-AD: Autonomous hyperspectral anomaly detection network based on fully convolutional autoencoder ".

<p align="center">
  <a href="https://github.com/TulioOParreiras/ProjectTemplate/issues">
    </a>
    <img src="https://github.com/lzz11834/SGIDN/blob/master/imgs/Chapter.3-Fig.3.png" />
	<br>
	<br>
</p

Abstract

Hyperspectral anomaly detection is aimed at detecting observations that differ from their surroundings, and is an active area of research in hyperspectral image processing. Recently, autoencoders (AEs) have been applied in hyperspectral anomaly detection; however, the existing AE-based methods are complicated and involve manual parameter setting and pre-processing and/or post-processing procedures. In this paper, an autonomous hyperspectral anomaly detection network (Auto-AD) is proposed, in which the background is reconstructed by the network and the anomalies appear as reconstruction errors. Specifically, through a fully convolutional AE with skip connections, the background can be reconstructed while the anomalies are difficult to reconstruct, since the anomalies are relatively small compared to the background and have a low probability of occurring in the image. To further suppress the anomaly reconstruction, an adaptive-weighted loss function is designed, where the weights of potential anomalous pixels with large reconstruction errors are reduced during training. As a result, the anomalies have a higher contrast with the background in the map of reconstruction errors. The experimental results obtained on a public airborne dataset and two unmanned aerial vehicle-borne hyperspectral datasets confirm the effectiveness of the proposed Auto-AD method. 

## Citation and Contact

If you use our work, please also cite the paper:

```
@ARTICLE{9382262,
  author={Wang, Shaoyu and Wang, Xinyu and Zhang, Liangpei and Zhong, Yanfei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Auto-AD: Autonomous Hyperspectral Anomaly Detection Network Based on Fully Convolutional Autoencoder}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2021.3057721}}
```


## Requirements

Conda and Pytorch. 

## Features
* End-to-end anomaly detection framework.
* The network reconstructs the background and anomalies appear as reconstruction errors.


## Running experiments


```
 cd <path>
 # activate your virtual environment
 conda activate your_env_name
 # run experiment
 python AD_main_loop.py
```


## Demo
We provide a demo in ‘./demo’ using the HYDICE dataset. 

The WHU-Hi-Park and WHU-Hi-Station datasets in our paper can be downloaded in the following page:




## License
The copyright belongs to Intelligent Data Extraction, Analysis and Applications of Remote Sensing (RSIDEA) academic research group, State Key Laboratory of Information Engineering in Surveying, Mapping, and Remote Sensing (LIESMARS), Wuhan University. This program is for academic use only. For commercial use, please contact Professor. Zhong (zhongyanfei@whu.edu.cn). The homepage of RSIDEA is: http://rsidea.whu.edu.cn/.

## Acknowledgement

The authors would like to thank the authors of “Deep image prior” and “Deep hyperspectral prior: denoising, inpainting, super-resolution” for sharing their codes. 
