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



