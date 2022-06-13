# Weighted Entropic Associative Memories:  A Case Study on Phonetic Representation and Learning
This repository contains the procedures to replicate the experiments presented in the paper

Pineda, Luis A. & Rafael Morales (under review). “Weighted Entropic Associative Memory: A Case Study on Phonetic Representation and Learning”.

The code was written in Python 3, using the Anaconda Distribution, and was run on two computers:

1.  Desktop computer with the following specifications:
   * CPU: Intel Core i7-6700 at 3.40 GHz
   * GPU: Nvidia GeForce GTX 1080
   * OS: Ubuntu 16.04 Xenial
   * RAM: 64GB
2. Server with 150 Xeon Gold with 22 cores (using one CPU), and an NVIDIA Tesla P100 graphic card.

## Requeriments

The following libraries need to be installed beforehand:
* docopt
* joblib
* matplotlib
* numpy
* pandas
* python_speech_features
* scipy
* seaborn
* sklearn
* soundfile
* tensorflow / tensorflow-gpu

## Data

The DIMEx100 Corpus for Mexican Spanish:

* Pineda, L. A. et al. The Corpus DIMEx100: Transcription and Evaluation. *Lang. Resour. Eval. 44*, 347–370 (2010). URL https://turing.iimas.unam.mx/~luis/DIME/CORPUS-DIMEX.html

The CIEMPIESS Corpus for Mexican Spanish:

* Mena, C. D. H. & Camacho, A. H. Ciempiess: A New Open-Sourced Mexican Spanish Radio Corpus. In  N. C. C. et al.
  (eds.) Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14) (European
  Language Resources Association (ELRA), Reykjavik, Iceland, 2014). URL http://www.ciempiess.org/

## Use

To see how to use the code, just run the following command in the source directory

```shell
python eam.py --help
```



## License

Copyright [2022] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda, and Rafael Morales Gamboa.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
