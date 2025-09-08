# DucDiff

This repository provides a reference implementation of **DucDiff**, as described in the paper:
[DucDiff: Dual-consistent Diffusion for Uncertainty-aware Information Diffusion Prediction](https://ieeexplore.ieee.org/abstract/document/11123747)

## 🔧 Dependencies

Install the required dependencies using [Anaconda](https://www.anaconda.com/). Follow these steps:

```bash
# Create a virtual environment
conda create --name DucDiff python=3.10

# Activate the virtual environment
conda activate DucDiff

# Install PyTorch from the official PyTorch repository (CUDA 11.8)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Dataset

We provide the twitter dataset in our repository, if you want get other datasets, you can find them in our paper, or you can send email to us, we are pleased to offer you other datasets.

## 🚀 Usage

Here we provide the implementation of DucDiff along with Android dataset.

- To train and evaluate on Android:

```python
python run.py
```

More running options are described in the codes, e.g., `-data_name= Android`

## 📁 Folder Structure

DucDiff

```
└── data: # The file includes datasets
    ├── android
       ├── cascades.txt       # original data
       ├── cascadetrain.txt   # training set
       ├── cascadevalid.txt   # validation set
       ├── cascadetest.txt    # testing data
       ├── edges.txt          # social network
       ├── idx2u.pickle       # idx to user_id
       ├── u2idx.pickle       # user_id to idx

└── Constants.py:
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
└── Optim.py:          # Optimization.
...
```

## 📝 Cite

If you find our paper & code are useful for your research, please consider citing us 😘:

```bibtex
@article{zhong2025ducdiff,
  title={DucDiff: Dual-consistent Diffusion for Uncertainty-aware Information Diffusion Prediction},
  author={Zhong, Ting and Ye, Wenxue and Li, Shichong and Liu, Yang and Cheng, Zhangtao and Zhou, Fan and Chen, Xueqin},
  journal={IEEE Transactions on Big Data},
  year={2025},
  publisher={IEEE}
}
```

## 📧 Contact

For any questions please open an issue or drop an email to: `wenxue.ye at outlook.com`
