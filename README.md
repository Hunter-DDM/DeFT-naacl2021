# DeFT-NAACL2021

* Paper: Decompose, Fuse and Generate: A Formation-Informed Method for Chinese Definition Generation ([URL](https://www.aclweb.org/anthology/2021.naacl-main.437.pdf))

## Requirements
* Pytorch (1.6.0)
* numpy (1.19.4)
* nltk (3.5)
* jieba (0.42.1)
* gensim (3.8.3)

## Prepare Required Data

Download our formation-informed DG dataset ([URL will be released later]()) and put it into `./data/morpheme/`

Download [gigaword Chinese word vectors](https://fasttext.cc/docs/en/crawl-vectors.html) and put it into `./data/morpheme/`

## Run Experiments

```
bash run_morphemes.sh
```
You can change the options defined in `main.py` to train models under different configurations. 

## Trained checkpoints

We will release our trained checkpoints ([URL will be released later]()) later to help reproduce our reported results. 

## Citation

If you use this code for your research, please kindly cite our NAACL-2021 paper:
```
@inproceedings{zheng2021morpheme,
  title={Decompose, Fuse and Generate: A Formation-Informed Method for Chinese Definition Generation},
  author={Zheng, Hua and Dai, Damai and Li, Lei and Liu, Tianyu and Sui, Zhifang and Chang, Baobao and Liu, Yang},
  booktitle={NAACL 2021, to specify later}, 
  pages={to specify later},
  year={2021}
}
```

## Acknowledgements

We refer to [ishiwatari-naacl2019](https://github.com/shonosuke/ishiwatari-naacl2019) to develop our project (DeFT). 

## Contact

Damai Dai: daidamai@pku.edu.cn
Hua Zheng: zhenghua@pku.edu.cn
