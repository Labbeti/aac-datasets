<!-- # -*- coding: utf-8 -*- -->

<div align="center">

# Automated Audio Captioning datasets for Pytorch

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.1-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<a href="https://github.com/Labbeti/aac_datasets/actions"><img alt="Build" src="https://img.shields.io/github/workflow/status/Labbeti/aac_datasets/Python%20package%20using%20Pip/main?style=for-the-badge&logo=github"></a>

Automated Audio Captioning Unofficial datasets source code for **AudioCaps** [1], **Clotho** [2], and **MACS** [3], designed for Pytorch.

</div>

## Installation

```bash
pip install git+https://github.com/Labbeti/aac_datasets
```
or clone the repository :
```bash
git clone https://github.com/Labbeti/aac_datasets
pip install -e aac_datasets
```

## Examples

### Create Clotho dataset

```python
from aac_datasets import Clotho

dataset = Clotho(root=".", subset="dev", download=True)
audio, captions, *_ = dataset[0]
# audio: Tensor of shape (n_channels=1, audio_max_size)
# captions: list of str captions
```

### Build Pytorch dataloader with MACS

```python
from torch.utils.data.dataloader import DataLoader
from aac_datasets import MACS
from aac_datasets.utils import BasicCollate

dataset = MACS(root=".", download=True)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=BasicCollate())

for audio_batch, captions_batch in dataloader:
    # audio_batch: Tensor of shape (batch_size=4, n_channels=2, audio_max_size)
    # captions_batch: list of list of str captions
    ...
```

## Datasets stats
Here is the statistics for each dataset :

| | AudioCaps | Clotho | MACS |
|:---:|:---:|:---:|:---:|
| Subset(s) | train, val, test | dev, val, eval, test, analysis | full |
| Sample rate | 32000 | 44100 | 48000 |
| Estimated size | 43GB | 27GB | 13GB |
| Audio source | AudioSet (youtube) | Freesound | TAU Urban Acoustic Scenes 2019 |

Here is the **train** subset statistics for each dataset :

| | AudioCaps/train | Clotho/dev | MACS/full |
|:---:|:---:|:---:|:---:|
| Nb audios | 49838 | 3840 | 3930 |
| Total audio duration | 136.6h<sup>1</sup> | 24.0h | 10.9h |
| Audio duration range | 0.5-10s | 15-30s | 10s |
| Nb captions per audio | 1 | 5 | 2-5 |
| Nb captions | 49838 | 19195 | 17275 |
| Total nb words<sup>2</sup> | 402482 | 217362 | 160006 |
| Nb words range<sup>2</sup> | 1-52 | 8-20 | 5-40 |

<sup>1</sup> This duration is estimated on the total duration of 46230/49838 files of 126.7h.

<sup>2</sup> The sentences are cleaned (lowercase+remove punctuation) and tokenized using the spacy tokenizer to count the words.

## Requirements
### Python packages

The requirements are automatically installed when using pip on this repository.
```
torch >= 1.10.1
torchaudio >= 0.10.1
py7zr >= 0.17.2
pyyaml >= 6.0
tqdm >= 4.64.0
```

### External requirements (AudioCaps only)

The external requirements needed to download **AudioCaps** are **ffmpeg** and **youtube-dl**.
These two programs can be download on Ubuntu using `sudo apt install ffmpeg youtube-dl`.

You can also override their paths for AudioCaps:
```python
from aac_datasets import AudioCaps
AudioCaps.FFMPEG_PATH = "/my/path/to/ffmpeg"
AudioCaps.YOUTUBE_DL_PATH = "/my/path/to/youtube_dl"
_ = AudioCaps(root=".", download=True)
```

## Command line download
To download a dataset, you can use `download=True` argument in dataset construction.
However, if you want to download datasets separately, you can also use the following command :
```bash
python -m aac_datasets.download --root "./data" clotho --version "v2.1"
```

## References

[1] C. D. Kim, B. Kim, H. Lee, and G. Kim, “Audiocaps: Generating captions for audios in the wild,” in NAACL-HLT, 2019. Available: https://aclanthology.org/N19-1011/

[2] K. Drossos, S. Lipping, and T. Virtanen, “Clotho: An Audio Captioning Dataset,” arXiv:1910.09387 [cs, eess], Oct. 2019, Available: http://arxiv.org/abs/1910.09387

[3] F. Font, A. Mesaros, D. P. W. Ellis, E. Fonseca, M. Fuentes, and B. Elizalde, Proceedings of the 6th Workshop on Detection and Classication of Acoustic Scenes and Events (DCASE 2021). Barcelona, Spain: Music Technology Group - Universitat Pompeu Fabra, Nov. 2021. Available: https://doi.org/10.5281/zenodo.5770113
