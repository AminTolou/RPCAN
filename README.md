Single Image Super Resolution using Parallel Channel Attention Based on RCAN Method (RPCAN)
The  code is based on the EDSR (Enhanced Deep Super-Resolution) and RCAN (Residual Channel Attention Network) methods. 
https://github.com/sanghyun-son/EDSR-PyTorch
https://github.com/yulunzhang/RCAN
This implementation takes the foundations of those previous models and builds upon them.

This code is an implementation of the paper titled "Single Image Super Resolution using Parallel Channel Attention Based on RCAN Method (RPCAN)" which was recently submitted.

This paper deals with the problem of single image super-resolution. Recently, the channel attention mechanism in deep neural networks has shown great success in image super-resolution. 
One of the most common methods used in channel attention is the global average pooling. It forces the model to pay attention to the channels with higher average values. A key challenge 
is that the channel with higher average does not necessarily contain the most important information for super-resolution reconstruction. There are cases where a channel with low average 
value provides more details to enhance the input image. In this paper, we suggest to use the contrast of each channel besides the average as a criterion for channel attention. We expect
the contrast of channels with rich texture to be higher than other channels. Accordingly, we propose a parallel branch in the channel attention module, to emphasize either the global 
average or the contrast for reconstructing the super-resolution image. 
      The proposed method has been applied to five benchmark datasets considering three different scales. Evaluation of the results in terms of performance metrics as well as visual assessment
, shows that the proposed method produces better super-resolution images.

![image](https://github.com/AminTolou/RPCAN/assets/44254357/cab29c17-4763-4e9f-a72e-6e52dae2ced3)

![image](https://github.com/AminTolou/RPCAN/assets/44254357/44f8e60f-7eb1-4d7f-9b9d-60b8c89518e0)


![image](https://github.com/AminTolou/RPCAN/assets/44254357/a2c625c7-a8af-44c8-ac18-205a4e083a2e)


4.	Experimental Results
4.1 Datasets and Evaluation Metrics
In the field of Single Image Super-Resolution, there are six benchmark datasets: DIV2K[22], Set5 [23], Set14[24], BSD100 [25], Urban100 [26] and Manga109 [27]. Recent researchers mostly used the DIV2K
 dataset to train their models due to its extensive scale and image variety, and evaluated their models using SET5, SET14, BSD100, Urban100, and Manga109 datasets. For a fair comparison, we also adopt
the same. Specifically, we utilized DIV2K to train our model for image super-resolution. The Other benchmarking datasets has been exclusively employed for performance analysis.
4.2 Implementation Details
To implement the proposed method, a computer with 24 GB RAM, Core (TM) i7 CPU, 64-bit Win10 operating system, Nvidia GeForce 1050 4GB GPU, and the Pytorch library were utilized. Specifying the model
hyperparameters, there are 10 residual groups, 20 residual blocks per group, 64 channels per convolutional layer, and a reduction ratio equal to 16. The optimization uses Adam with an initial learning
rate set to 10-4. The learning rate is halved  after every 200 update steps across 1000 total training epochs. 
        It should be noted that due to the unavailability of a powerful processing system, the results are reported using a batch size of 1 (instead of 16). For a fair comparison, the RCAN method was
  	also implemented with a batch size of 1, and the results of the two methods with the same batch size are compared.

   ![image](https://github.com/AminTolou/RPCAN/assets/44254357/4c64b44f-77ae-41ae-8a03-098f0747b4d3)

############################################################################################
How to train the model
############################################################################################
1- Prepare training data
Download DIV2K training data

2- Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

3-  download bench mark data base that were used for Test 

4- Unpack the tar file to any place you want. Then, change the dir_data argument in src/option.py to the place where DIV2K images are located.

5-We recommend you to pre-process the images before training. This step will decode all png files and save them as binaries. Use --ext sep_reset argument on your first run. You can skip the decoding part and use saved binaries with --ext sep argument.

You can train code by yourself. All scripts are provided in the src/demo.sh. 

cd src       
sh demo.sh
###############################################################################################
Dependencies
################################################################################################
cffi==1.14.2
cmake==3.18.0
cycler==0.10.0
decorator==4.4.2
future==0.18.2
imageio==2.9.0
intel-openmp==2020.0.133
kiwisolver==1.1.0
matplotlib==3.0.3
mkl==2019.0
mkl-include==2019.0
networkx==2.4
numpy==1.18.5
Pillow==7.2.0
pycparser==2.20
pyparsing==2.4.7
python-dateutil==2.8.1
PyWavelets==1.1.1
PyYAML==5.3.1
scikit-image==0.15.0
scipy==1.4.1
six==1.15.0
 torch==2.1.0
 torchvision==0.16.0
 torchaudio==2.1.0
tqdm==4.19.9
typing==3.7.4.3
