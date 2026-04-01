<div align="center">

# A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparascopic Video


Maximilian Fehrentz*<sup>1,2,4</sup> · Nicolas Stellwag*<sup>2</sup> · Robert Wiebe<sup>2</sup> · Nicole Thorisch<sup>2</sup> · Fabian Grob<sup>2</sup> · Patrick Remerscheid<sup>2</sup> · Ken-Joel Simmoteit<sup>2</sup> · Benjamin D. Killeen<sup>1,4</sup> · Christian Heiliger<sup>3</sup> · Nassir Navab<sup>1,4</sup>

**<sup>1</sup>Computer Aided Medical Procedures, TU Munich**

**<sup>2</sup>TUM.ai**

**<sup>3</sup>University Hospital of Ludwig Maximilian University (LMU)
Munich**

**<sup>4</sup>Munich Center for Machine Learning**

### [Project Page](https://tum-ai.github.io/surg4d/) | [Paper](https://example.com) | [Dataset](https://github.com/tum-ai/surg4d)

</div>

---
Official implementation of "A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparascopic Video".

## Installation & Setup

### 1. Install pixi

Install the pixi package manger using the [official instructions](https://pixi.prefix.dev/latest/installation/).

_Note: This installs pixi into the user's home directory and python environments will later be placed in this project directory. So there should be no issues on compute clusters._


### 2. Clone repository

```bash
git clone --recurse-submodules git@github.com:tum-ai/surg4d.git
```

### 3. Setup python environment

```bash
# install conda and pypi packages
pixi install

# install custom packages and download checkpoints
pixi run setup

# optional: test importing key packages
pixi run test-install
```

### 4. Download dataset and annotations

Download the [CholecSeg8k dataset](https://www.kaggle.com/datasets/newslab/cholecseg8k) and our annotations:
```bash
pixi run download-cholecseg8k
pixi run download-benchmark-annotations
```

_Note: If you want to annotate your own queries, check out our [annotation tool repository](https://github.com/NicoStellwag/surg4d-benchmark)_.

## Usage

The pipeline is based on the configuration system hydra. Config files can be found in `conf/`. Check out the hydra [getting started](https://hydra.cc/docs/intro/) guide.

Run the pipeline using the following scripts:

```bash
# train segmentation model and create masks
pixi run python segment.py

# preprocess frames, masks, and annotations
pixi run python preprocess.py

# predict depth and pose
pixi run python extract_geometry.py

# create temporally consistent instances
pixi run python track_objects.py

# build 4d scene graphs
pixi run python extract_graphs.py

# predict benchmark queries
pixi run python evaluate_benchmark.py

# compute benchmark metrics
pixi run python compute_metrics.py
```

Or the whole pipeline, including all ablations:

```bash
bash ablate_all.sh
```

<!-- ## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fehrentz2025bridgesplat,
  title={BridgeSplat: Bidirectionally Coupled CT and Non-rigid Gaussian Splatting for Deformable Intraoperative Surgical Navigation},
  author={Fehrentz, Maximilian and Winkler, Alexander and Heiliger, Thomas and Haouchine, Nazim and Heiliger, Christian and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={44--53},
  year={2025},
  organization={Springer}
}
``` -->

## 📧 Contact

For questions, please open an issue or contact [maximilian.fehrentz@tum.de](mailto:maximilian.fehrentz@tum.de).