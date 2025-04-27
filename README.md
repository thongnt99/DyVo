# DyVo

Codebase for the paper:  
**"DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities"**  
(EMNLP 2024)

---

## Steps to Run the Code

### 1. Create Conda Environment and Install Dependencies

Create and activate the environment:
```bash
conda create --name lsr python=3.9.12
conda activate lsr
```

Install required packages:
```bash
pip install -r requirements.txt
```

---

### 2. Download Codebase and Data

#### 2.1 Clone the DyVo Repository
```bash
git clone https://github.com/thongnt99/DyVo
```

#### 2.2 Create a Data Directory
```bash
cd DyVo
mkdir dyvo_data
cd dyvo_data
```

#### 2.3 Download Data from Hugging Face

Make sure the Hugging Face CLI is installed:
```bash
pip install huggingface_hub
```

Then download the data:
```bash
huggingface-cli download lsr42/dyvo_data
```

**Note**:  
- You may need to log in to Hugging Face before downloading:
  ```bash
  huggingface-cli login
  ```
- The downloaded files will be cached locally. Refer to the Hugging Face CLI documentation for cache settings if needed.
- Recommended Python version: 3.7 or later.

---

### 3. Train and Evaluate a Model

Example command to start training:
```bash
python -m lsr.train +experiment=qmlp_dmlm_emlm_laque_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.001_entw_0.05.yaml training_arguments.fp16=True
```

- The list of experiment configuration files can be found in the `lsr/configs/experiment/` directory.

---

## Citing DyVo

If you find this repository helpful, please cite our paper:

```bibtex
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
