# Fast Abductive Learning by Similarity-based Consistency Optimization

This is the repository for holding the sample code of _[Fast Abductive Learning by Similarity-based Consistency Optimization](https://proceedings.neurips.cc/paper/2021/file/df7e148cabfd9b608090fa5ee3348bfe-Paper.pdf)_ in NeurIPS 2021.

This code is only tested in Linux environment.

## Environment Dependency

- Ubuntu 18.04
- Python 3.7
- PyTorch 1.7
- CuPy 8.3
- tqdm
- scikit-learn
- opencv-python

To create the above environment with [Anaconda](https://www.anaconda.com/products/distribution), you can run the following command (cudatoolkit=10.1 for old GPUs, cudatoolkit=11.3 for new GPUs / new drivers):

 (cudatoolkit=10.1)

```
conda create -n ablsim python=3.7 -y
conda activate ablsim
conda install -c conda-forge cupy cudatoolkit=10.1 -y
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch -y
pip install tqdm opencv-python scikit-learn matplotlib
python main_1_2.py --dataset 2ADD --images CIFAR
```

 (cudatoolkit=11.3)

```
conda create -n ablsim python=3.7 -y
conda activate ablsim
conda install -c conda-forge cupy cudatoolkit=11.3 -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install tqdm opencv-python scikit-learn matplotlib
python main_1_2.py --dataset 2ADD --images CIFAR
```

## Running Code

To reproduce the experiment results, simply run the following code:

Download the [Handwritten_Math_Symbols.zip](https://drive.google.com/file/d/1tItMQAxoqKW6C4wC3tTc0asPu6zD9v4V/view?usp=sharing) from google drive and unzip it:

```
unzip Handwritten_Math_Symbols.zip -d data
```

- MNIST (CIFAR-10) Addition

  ```
  python main_1_2.py --dataset 2ADD --images handwritten 
  python main_1_2.py --dataset 2ADD --images CIFAR 
  ```

- Handwritten Formula Recognition

  ```
  python main_1_2.py --dataset HWF --images handwritten
  python main_1_2.py --dataset HWF --images CIFAR 
  ```

- CIFAR-10 Decimal Equation Decipherment

  Download the [images.zip](https://drive.google.com/file/d/15SvSF-mVLMjAKD5019IFGL9DgDtsLFQg/view?usp=sharing) and [ssl_mode.zip](https://drive.google.com/file/d/1dRdOiJnYqFpibypepEdI-v5lT5CdmwBf/view?usp=sharing) from google drive and unzip it:
  
  ```
  unzip images.zip -d data
  unzip ssl_model.zip
  python main_3.py --images CIFAR
  ```

To view or change the hyperparameters, please refer to the *arg_init()* function in the code.

## Reference

```
@incollection{ablsim2021huang,
	author = {Huang, Yu-Xuan and Dai, Wang-Zhou and Cai, Le-Wen and Muggleton, Stephen H and Jiang, Yuan},
	booktitle = {Advances in Neural Information Processing Systems 34},
	pages = {26574--26584},
	title = {Fast Abductive Learning by Similarity-based Consistency Optimization},
	year = {2021}
}
```
