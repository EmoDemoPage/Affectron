<div align="center">

## Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations

</div>
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/logo.gif">
  <img src="Figures/logo.gif" alt="Affectron Logo" width="800">
</picture>

</div>

<div align="center">

<a href="https://emodemopage.github.io/Affectron-Demo/" target="_blank">
  <img src="https://img.shields.io/badge/🎧%20Demo-Listen%20Here-8A2BE2?style=for-the-badge" />
</a>

</div>

## 📰 News
- **2025-12-21**: We officially released **Affectron**, along with an interactive demo page showcasing affective and contextually aligned nonverbal vocalizations in emotional speech synthesis.

## TODO
- [x] Codebase upload
- [x] Environment setup
- [ ] Inference demo
- [ ] Training guidance
- [ ] Dataset and training manifest

## Introduction
<div align="center">
  <img src="https://github.com/user-attachments/assets/201d8a7a-fb9b-4939-8c4a-d283779d4846" width="75%">
  <br>
  <em>Overall framework of Affectron.</em>
</div>

<br>

Emotional speech synthesis benefits greatly from nonverbal vocalizations (NVs), such as laughter and sighs, which convey affect beyond words. However, NVs are often underrepresented due to limited data availability and reliance on proprietary resources or NV detectors.

We propose **Affectron**, a framework that generates affectively and contextually aligned NVs using NV-augmented training on a small-scale open corpus. 


## Environment setup
```bash
conda create -n voicecraft python=3.9.16
conda activate voicecraft

pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
apt-get install ffmpeg # if you don't already have ffmpeg installed
apt-get install espeak-ng # backend for the phonemizer installed below
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

## Inference Examples


## Training


## Pretrained checkpoints

## Acknowledgements
**Our codes are based on the following repos:**
* [VoiceCraft](https://github.com/jasonppy/VoiceCraft/tree/master)
