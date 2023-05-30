
Pytorch implementation of our Multiresolution Learning Approach with a Hybrid Transformer-CNN Model framework.

https://github.com/seriee/Multiresolution-Learning-based-Hybrid-Transformer-CNN-Model-for-Anatomical-Landmark-Detection

## Abstract:
Accurate localization of anatomical landmarks has a critical role in clinical diagnosis, treatment planning, and research. Most existing
deep learning methods for anatomical landmark localization rely on heatmap regression-based learning, which generates label representations
as 2D Gaussian distributions centered at the labeled coordinates of each of the landmarks and integrates them into a single spatial resolution
heatmap. However, the accuracy of this method is limited by the resolution of the heatmap, which restricts its ability to capture finer details. In
this study, we introduce a multiresolution heatmap learning strategy that enables the network to capture semantic feature representations precisely
using multiresolution heatmaps generated from the feature representations at each resolution independently, resulting in improved localization
accuracy. Moreover, we propose a novel network architecture called hybrid transformer-CNN (HTC), which combines the strengths of both
CNN and vision transformer models to improve the network’s ability to effectively extract both local and global representations. We evaluated
our approach on the numerical XCAT 2D projection images and two public X-ray landmark detection benchmark datasets. Extensive experiments
demonstrated that our approach outperforms state-of-the-art deep learning-based anatomical landmark localization networks.

## Hybrid Transformer-CNN Model with Multiresolution Learning Approach
<div align="center">
  <img src="resources/Multiresolution_Learning_HTC.PNG"/>
</div>


## Hybrid Transformer-CNN Architecture
<div align="center">
  <img src="resources/HTC_architecture.PNG", width=700/>
</div>
