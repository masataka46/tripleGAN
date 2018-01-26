# Sample code for Triple GAN

## Discription
This software is a python implementation of Triple GAN using Tensorflow.  

See the paper.
https://arxiv.org/abs/1703.02291

## Requirement
I only confirm the operation under this environment
1.  python 3.5.3
2.  tensorflow-gpu 1.5.0rc1
3.  pillow 5.0.0
4.  numpy 1.14.0

## Usage
First of all, prepare mnist data as npy form.  See my tripleGAN code in chainer.  
https://github.com/masataka46/tripleGAN_chainer  
Then, you can get mnist_train_img.npy, mnist_train_label.npy, mnist_test_img.npy and mnist_test_label.npy.  

To train this model, do `python train_tripleGAN.py`

## License
MIT 
