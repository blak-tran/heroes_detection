## Installation

Provide step-by-step instructions on how to install and set up your project. Include any dependencies that need to be installed and any configuration steps. For example:

```bash
git clone https://github.com/blak-tran/heroes_detection.git

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

# Retrain the model.
python train.py

# Code inference
python inference.py
