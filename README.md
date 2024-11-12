# DyVo

Codebase for the paper "DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities" EMNLP 2024 

### Steps to run the code: 

1.  Create conda environment and install dependencies:

Create conda environment: 
```bash
conda create --name lsr python=3.9.12
conda activate lsr
```
Install dependencies: 
```bash
pip install -r requirements.txt
```
2. Train and evaluatate a model 

```bash 
python -m lsr.train +experiment=qmlp_dmlm_emlm_laque_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.001_entw_0.05.yaml training_arguments.fp16=True 
```

The list of experiment configurations could be found inside the `lsr/configs/experiment` directory. 

### Citing and Authors

If you find this repository helpful, feel free to cite our paper

```
@inproceedings{nguyen-etal-2024-dyvo,
    title = "DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities",
    author = "Nguyen, Thong  and
      Chatterjee, Shubham  and
      MacAvaney, Sean  and
      Mackie, Iain  and
      Dalton, Jeff  and
      Yates, Andrew",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024"
}
```
