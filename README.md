# Fast Abductive Learning by Similarity-based Consistency Optimization

This is the repository for holding the sample code of _Fast Abductive Learning by Similarity-based Consistency Optimization_ submitted in NeurIPS 2021.

This code is only tested in Linux environment.

## Environment Dependency

- Ubuntu 18.04
- Python 3.7
- PyTorch 1.7
- CuPy 8.3
- tqdm
- scikit-learn
- opencv-python

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

