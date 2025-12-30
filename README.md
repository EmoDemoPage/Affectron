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

<br>

## 📰 News
- **2025-12-21**: We officially released **Affectron**, along with an interactive demo page showcasing affective and contextually aligned nonverbal vocalizations in emotional speech synthesis.
- **2025-12-30**: Added comprehensive **Training** and **Inference** guidance to facilitate reproducibility and ease of use.

<br>

## ⭐ TODO
- [x] Codebase upload
- [x] Environment setup
- [x] Training guidance
- [x] Inference guidance
- [ ] Pretrained checkpoints

<br>

## Introduction
<div align="center">
  <img src="https://github.com/user-attachments/assets/201d8a7a-fb9b-4939-8c4a-d283779d4846" width="75%">
  <br>
  <em>Overall framework of Affectron.</em>
</div>

<br>

Emotional speech synthesis benefits greatly from nonverbal vocalizations (NVs), such as laughter and sighs, which convey affect beyond words. However, NVs are often underrepresented due to limited data availability and reliance on proprietary resources or NV detectors.

We propose **Affectron**, a framework that generates affectively and contextually aligned NVs using NV-augmented training on a small-scale open corpus. 

<br>

---

## 0. Environment setup
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

<br>

---

## 1. Training

This section describes the full training pipeline for Affectron, including dataset preparation, feature extraction, and model training.

<br>

### Step 1) Download EARS dataset and split Verbal / NV recordings

Download the EARS dataset following the official instructions: 
- https://github.com/facebookresearch/ears_dataset

After downloading, split the recordings into **verbal** and **nonverbal vocalization (NV)** subsets as required by the training pipeline.

<br>

### Step 2) Encodec encoding and phoneme extraction

We provide a preprocessing script that:
- loads utterances and their transcripts,
- encodes utterances into discrete codes using **Encodec**,
- converts transcripts into **phoneme sequences**,
- and builds a phoneme vocabulary (`vocab.txt`).

Run the following command:

```bash
conda activate voicecraft
export CUDA_VISIBLE_DEVICES=0
cd ./data

python phonemize_encodec_encode_hf.py \
  --dataset_size xs \
  --download_to path/to/store_huggingface_downloads \
  --save_dir path/to/store_extracted_codes_and_phonemes \
  --encodec_model_path path/to/encodec_model \
  --mega_batch_size 120 \
  --batch_size 32 \
  --max_len 30000
```
#### Encodec model
Use the **same Encodec model as the VoiceCraft baseline**:
- https://huggingface.co/pyp1/VoiceCraft \
This model is trained on **GigaSpeech XL**, has **56M parameters**, and uses **4 codebooks**, each with **2048 codes**.

<br>

### Step 3) Model training
Start training with:
```bash
sh Train_Affectron_TTSbase.sh
```

Before running the script, make sure to configure the following variables according to your environment:
- `dataset`
- `model_name`
- `exp_name`
- `exp_root`
- `dataset_dir`
- `load_model_from`
Training logs and checkpoints will be saved under `exp_root`.

<br>

---

## 2. Inferece

### Step 1) Create a manifest file

Create a meta file under the `./manifest` directory. \
Each line is **tab-separated** and consists of:
1. Reference audio path
2. Text (reference transcript + target generation text)
3. Utterance ID
4. Prompt audio length (seconds)
5. Prompt start time (seconds)

Example:
```bash
/dataset/EARS_final/VVNVs/p005_emo_adoration_sentences_0.wav	You're just the sweetest person I know, and I'm so happy to call you my friend. I had the best time with you.	p005_emo_adoration_sentences_0	5.9	0.0
```
⚠️ Important (same as the VoiceCraft baseline): \
Ensure **(prompt length + generation length) ≤ 16 seconds**. \
Due to limited compute, utterances longer than **16 seconds** were excluded during training.

<br>

### Step 2) Run inference
```bash
sh Inference_Affectron_TTSbase.sh
```
Before running, set:
- `model_name`
- `exp_dir`
- `output_dir` 
Generated audio files will be saved under `output_dir`.

<br>

--- 
## 3. Pretrained checkpoints

To preserve anonymity and avoid download tracking during the review process, pretrained checkpoints will be released **after the review is completed**.

<br>

---
## 4. Acknowledgements
**Our codes are based on the following repos:**
* [VoiceCraft](https://github.com/jasonppy/VoiceCraft/tree/master)
