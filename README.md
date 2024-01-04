# Beyond Image-Text Matching: Verb Understanding in Multimodal Transformers Using Guided Masking

In the paper Beyond Image-Text Matching: Verb Understanding in Multimodal Transformers Using Guided Masking, all experiments were carried out using implemen tations of the ViLBERT, LXMERT, UNITER and VisualBERT models1 provided by VOLTA (Visiolinguistic Transformer Architectures) ([Bugliarello et al., 2021](https://arxiv.org/abs/2011.15124)) [[Original code](https://github.com/e-bug/volta)], a PyTorch implementation of a unified mathematical framework of currently proposed V&L BERTs. The following instructions are a copy from VOLTA github, on how to get the code running. We made changes tosome of the files to complete the necessary experiments, however the initial approach to start is the same.

## VOLTA: Visiolinguistic Transformer Architectures

This is the implementation of the framework described in the paper:
> Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki and Desmond Elliott. [Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs](https://arxiv.org/abs/2011.15124). _Transactions of the Association for Computational Linguistics_ 2021; 9 978–994.

We provide the code for reproducing our results, preprocessed data and pretrained models.

## News
- 09-2021: Added code for `cross-modal ablation` ([Frank et al., EMNLP 2021](https://arxiv.org/pdf/2109.04448.pdf)) [[Original code](https://github.com/e-bug/cross-modal-ablation)]


## Repository Setup

You can clone this repository with submodules included issuing: <br>
`git clone git@github.com:e-bug/volta`

1\. Create a fresh conda environment, and install all dependencies.
```text
conda create -n volta python=3.6
conda activate volta
pip install -r requirements.txt
```

2\. Install PyTorch
```text
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch
```

3\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

4\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd tools/refer; make
```

5\. Install this codebase as a package in this environment.
```text
python setup.py develop
```


## Data

Check out [`data/README.md`](data/README.md) for links to preprocessed data and data preparation steps.


## Models

Check out [`MODELS.md`](MODELS.md) for links to pretrained models and how to define new ones in VOLTA.

Model configuration files are stored in [config/](config). 


## Training and Evaluation

We provide sample scripts to evaluate models in [examples/](examples).

Task configuration files are stored in [config_tasks/](config_tasks).


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@article{bugliarello-etal-2021-multimodal,
    author = {Bugliarello, Emanuele and Cotterell, Ryan and Okazaki, Naoaki and Elliott, Desmond},
    title = "{Multimodal Pretraining Unmasked: {A} Meta-Analysis and a Unified Framework of Vision-and-Language {BERT}s}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {978-994},
    year = {2021},
    month = {09},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00408},
    url = {https://doi.org/10.1162/tacl\_a\_00408},
}
```


## Acknowledgement

Our codebase heavily relies on these excellent repositories:
- [VOLTA](https://github.com/e-bug/volta)
- [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task)
- [vilbert_beta](https://github.com/jiasenlu/vilbert_beta)
- [lxmert](https://github.com/airsplay/lxmert)
- [VL-BERT](https://github.com/jackroos/VL-BERT)
- [visualbert](https://github.com/uclanlp/visualbert)
- [UNITER](https://github.com/ChenRocks/UNITER)
- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
- [transformers](https://github.com/huggingface/transformers)
