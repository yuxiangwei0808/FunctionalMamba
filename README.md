# [RECOMB'25] Hierarchical Spatio-Temporal State-Space Modeling for fMRI Analysis

Functional spatiotemporal Mamba (FST-Mamba) for functional network connectivity (FNC)-based brain modeling. [[Paper](https://arxiv.org/abs/2408.13074)]

<div align=left>
<img src=pics/framework.png width=90% />
</div>

## 🌟 Requirements
1. Clone this repository and navigate to RULE folder
```bash
git clone https://github.com/yuxiangwei0808/FunctionalMamba
cd FunctionalMamba
```

2. Install Package: Create conda environment

```Shell
conda env create -f environment.yml
conda activate fmamba
pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare data. Download the HCP (HCP-Rest), UKBiobank, and ADNI datasets (ADNI2, ADNI3) from the official datasets. Extract independent components or ROI, then compute the functional connectivity.

4. Run example script `slurm_script.sh`