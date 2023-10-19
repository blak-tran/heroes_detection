## Installation

Provide step-by-step instructions on how to install and set up your project. Include any dependencies that need to be installed and any configuration steps. For example:

```bash
git clone https://github.com/blak-tran/heroes_detection.git

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

# Download dataset
gdown https://drive.google.com/uc?id=1h_ugiooEMf0fepl6y6X4EbATM3Njl-ci

# Download checkpoint
gdown https://drive.google.com/uc?id=1cbQhKQhE7kdymDGomoEQaSgVOdj9Ojv9

# Pull data from League_of_Legends:_Wild_Rift
python pull_data.py

# Retrain the model.
python train.py

# Code inference
python inference.py
