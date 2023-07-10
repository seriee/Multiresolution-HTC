# Anatomical Landmark Detection Using a Multiresolution Learning Approach with a Hybrid Transformer-CNN Model (MICCAI 2023)


**This is the official pytorch implementation repository of our Anatomical Landmark Detection Using a Multiresolution Learning Approach with a Hybrid Transformer-CNN Model framework**: https://github.com/seriee/Multiresolution-HTC.git

## Abstract
Accurate localization of anatomical landmarks has a critical role in clinical diagnosis, treatment planning, and research. Most existing deep learning methods for anatomical landmark localization rely on heatmap regression-based learning, which generates label representations as 2D Gaussian distributions centered at the labeled coordinates of each of the landmarks and integrates them into a single spatial resolution heatmap. However, the accuracy of this method is limited by the resolution of the heatmap, which restricts its ability to capture finer details. In this study, we introduce a multiresolution heatmap learning strategy that enables the network to capture semantic feature representations precisely using multiresolution heatmaps generated from the feature representations at each resolution independently, resulting in improved localization accuracy. Moreover, we propose a novel network architecture called hybrid transformer-CNN (HTC), which combines the strengths of both CNN and vision transformer models to improve the network's ability to effectively extract both local and global representations. 
Extensive experiments demonstrated that our approach outperforms state-of-the-art deep learning-based anatomical landmark localization networks on the numerical XCAT 2D projection images and two public X-ray landmark detection benchmark datasets.

## Hybrid Transformer-CNN (HTC) with Multiresolution Learning
<div align="center">
  <img src="resources/Multiresolution_learning_HTC.png"/>
</div>


## Hybrid Transformer-CNN (HTC) Architecture
<div align="center">
  <img src="resources/Hybrid_Transformer_CNN.png", width=600/>
</div>

## Dataset
- We have used the following datasets:
  - 4D XCAT Head CBCT dataset: Segars, W.P., Sturgeon, G., Mendonca, S., Grimes, J., Tsui, B.M.: 4d xcat phantom for multimodality imaging research. Medical physics 37(9), 4902–4915 (2010)
  - ISBI2023 challenge dataset: Anwaar Khalid, M., Zulfiqar, K., Bashir, U., Shaheen, A., Iqbal, R., Rizwan, Z., Rizwan, G., Moazam Fraz, M.: Cepha29: Automatic cephalometric landmark detection challenge 2023. arXiv e-prints pp. arXiv–2212 (2022)
  - Hand X-ray dataset: Payer, C., ˇStern, D., Bischof, H., Urschler, M.: Integrating spatial configuration into heatmap regression based CNNs for landmark localization. Medical Image Analysis 54, 207–219 (2019)
  
## Prerequesites
- Python 3.7
- MMpose 0.23

## Usage

- Input: .PNG images and JSON file
- Output: 2D landmark coordinates

- **Train**
  - To train our HTC model with a multiresolution learning approach, run the following command:
  ```
  sh train.sh
  ```
  - train.sh should include the following information:
  ```
  CUDA_VISIBLE_DEVICES=gpu_ids PORT=PORT_NUM ./tools/dist_train.sh \
  config_file_path num_gpus
  ```

- **Test**
  - To test the trained HTC model, run the following command:
  ```
  sh test.sh
  ```
  - test.sh should include the following information:
  ```
  CUDA_VISIBLE_DEVICES=gpu_id PORT=29504 ./tools/dist_test.sh config_file_path \
      model_weight_path num_gpus \
      # For evaluation of the Head XCAT dataset, use:
      --eval 'MRE_h','MRE_std_h','SDR_2_h','SDR_2.5_h','SDR_3_h','SDR_4_h'
      # For evaluation of ISBI2023 and Hand X-ray dataset, use:
      # --eval 'MRE_i2','MRE_std_i2','SDR_2_i2','SDR_2.5_i2','SDR_3_i2','SDR_4_i2'
  ```

## Citation 
If you find this code useful for your research, please kindly cite our paper. The citation of our paper will be updated soon.
