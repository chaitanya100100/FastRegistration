# FastRegistration
This repository contains code for the following paper:
```
Fast Registration of Photorealistic Avatars for VR Facial Animation  
Chaitanya Patel, Shaojie Bai, Te-Li Wang, Jason Saragih, and Shih-En Wei  
ECCV 2024
```
[[Project Website](https://chaitanya100100.github.io/FastRegistration/)]
[[Ava-256 Dataset](https://about.meta.com/realitylabs/codecavatars/ava256/)]

## Citation
Cite us if you use our model, code or data:
```
@inproceedings{patel2024fastregistration,
  title = {Fast Registration of Photorealistic Avatars for VR Facial Animation},
  author = {Patel, Chaitanya and Bai, Shaojie and Wang, Te-Li and Saragih, Jason and Wei, Shih-En},
  booktitle = {European Conference on Computer Vision ({ECCV})},
  year = {2024},
}
```

### Ava-256 Dataset
See the Ava-256 dataset page for more details. Note that the released dataset consists of assets/data of a similar number of subjects, although not exactly identical to those used in the paper, due to legal reasons.

### Code to train on Ava-256 dataset
Coming soon...

### Legacy model code
We provide model code in `legacy` directory which we used to train on internal dataset. Although it is not runnable on public dataset yet, it contains code for core modules. In particular, `legacy/sdm_module.py` and `legacy/st_module.py` contains the code for iterative refinement module (F) and style transfer module (S) respectively. `legacy/model/` contains implementation of network layers.