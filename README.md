This directory contains source code for reproducting the results in the ICML
publication (Yingxiang Yang, Zhihan Xiong, Tianyi Liu, Taiqing Wang, Chong Wang. Fourier learning with cyclical data. ICML2022). The repository contains two experiments:

1. the toy example, and 
2. the experiment on Twitter Sentiment-140 dataset, including both source code and the original log files. 

The code of Sentiment-140 is largely based on the existing repository:
https://github.com/google-research/federated/tree/master/semi_cyclic_sgd (Eichner, Koren, McMahan, Srebro, Talwar. Semi-Cyclic Stochastic Gradient Descent. ICML2019), with some modifications to reduce run time and adapt to the experiment settings of the Fourier learning paper. We sincerely appreciate the availability of the original code, and would encourage the readers to cite both papers when using the code for future experiment.

@inproceedings{eichner2019semi,
  title={Semi-cyclic stochastic gradient descent},
  author={Eichner, Hubert and Koren, Tomer and McMahan, Brendan and Srebro, Nathan and Talwar, Kunal},
  booktitle={International Conference on Machine Learning},
  pages={1764--1773},
  year={2019},
  organization={PMLR}
}

@inproceedings{yang2022fourier,
  title={Fourier learning with cyclical data},
  author={Yang, Yingxiang and Xiong, Zhihan and Liu, Tianyi and Wang, Taiqing and Wang, Chong},
  booktitle={International Conference on Machine Learning},
  pages={25280--25301},
  year={2022},
  organization={PMLR}
}
