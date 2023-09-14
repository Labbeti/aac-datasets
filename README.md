<!-- # -*- coding: utf-8 -*- -->

<div align="center">

# Audio Captioning datasets for PyTorch

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<a href="https://github.com/Labbeti/aac-datasets/actions"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/Labbeti/aac-datasets/python-package-pip.yaml?branch=main&style=for-the-badge&logo=github"></a>
<a href='https://aac-datasets.readthedocs.io/en/stable/?badge=stable'>
    <img src='https://readthedocs.org/projects/aac-datasets/badge/?version=stable&style=for-the-badge' alt='Documentation Status' />
</a>

Audio Captioning unofficial datasets source code for **AudioCaps** [[1]](#audiocaps), **Clotho** [[2]](#clotho), **MACS** [[3]](#macs), and **WavCaps** [[4]](#wavcaps), designed for PyTorch.

</div>

## Installation
```bash
pip install aac-datasets
```

If you want to check if the package has been installed and the version, you can use this command:
```bash
aac-datasets-info
```

## Examples

### Create Clotho dataset

```python
from aac_datasets import Clotho

dataset = Clotho(root=".", download=True)
item = dataset[0]
audio, captions = item["audio"], item["captions"]
# audio: Tensor of shape (n_channels=1, audio_max_size)
# captions: list of str
```

### Build PyTorch dataloader with Clotho

```python
from torch.utils.data.dataloader import DataLoader
from aac_datasets import Clotho
from aac_datasets.utils import BasicCollate

dataset = Clotho(root=".", download=True)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=BasicCollate())

for batch in dataloader:
    # batch["audio"]: list of 4 tensors of shape (n_channels, audio_size)
    # batch["captions"]: list of 4 lists of str
    ...
```

## Datasets stats
Here is the statistics for each dataset :

| | AudioCaps | Clotho | MACS | WavCaps |
|:---:|:---:|:---:|:---:|:---:|
| Subsets | train, val, test | dev, val, eval, dcase_aac_test, dcase_aac_analysis, dcase_t2a_audio, dcase_t2a_captions | full | as, as_noac, bbc, fsd, fsd_nocl, sb |
| Sample rate (kHz) | 32 | 44.1 | 48 | 32 |
| Estimated size (GB) | 43 | 53 | 13 | 941 |
| Audio source | AudioSet | FreeSound | TAU Urban Acoustic Scenes 2019 | AudioSet, BBC Sound Effects, FreeSound, SoundBible |

For Clotho, the dev subset should be used for training, val for validation and eval for testing.

Here is the **train** subset statistics for AudioCaps, Clotho and MACS datasets :

| | AudioCaps/train | Clotho/dev | MACS/full |
|:---:|:---:|:---:|:---:|
| Nb audios | 49,838 | 3,840 | 3,930 |
| Total audio duration (h) | 136.6<sup>1</sup> | 24.0 | 10.9 |
| Audio duration range (s) | 0.5-10 | 15-30 | 10 |
| Nb captions per audio | 1 | 5 | 2-5 |
| Nb captions | 49,838 | 19,195 | 17,275 |
| Total nb words<sup>2</sup> | 402,482 | 217,362 | 160,006 |
| Sentence size<sup>2</sup> | 2-52 | 8-20 | 5-40 |

<sup>1</sup> This duration is estimated on the total duration of 46230/49838 files of 126.7h.

<sup>2</sup> The sentences are cleaned (lowercase+remove punctuation) and tokenized using the spacy tokenizer to count the words.

## Requirements

This package has been developped for Ubuntu 20.04, and it is expected to work on most Linux distributions.
### Python packages

Python requirements are automatically installed when using pip on this repository.
```
torch >= 1.10.1
torchaudio >= 0.10.1
py7zr >= 0.17.2
pyyaml >= 6.0
tqdm >= 4.64.0
huggingface-hub >= 0.15.1
numpy >= 1.21.2
```

### External requirements (AudioCaps only)

The external requirements needed to download **AudioCaps** are **ffmpeg** and **youtube-dl** (yt-dlp should work too).
These two programs can be download on Ubuntu using `sudo apt install ffmpeg youtube-dl`.

You can also override their paths for AudioCaps:
```python
from aac_datasets import AudioCaps
dataset = AudioCaps(
    download=True,
    ffmpeg_path="/my/path/to/ffmpeg",
    ytdl_path="/my/path/to/youtube_dl",
)
```

## Download datasets
To download a dataset, you can use `download` argument in dataset construction :
```python
dataset = Clotho(root=".", subset="dev", download=True)
```
However, if you want to download datasets from a script, you can also use the following command :
```bash
aac-datasets-download --root "." clotho --subsets "dev"
```

## References
#### AudioCaps
[1] C. D. Kim, B. Kim, H. Lee, and G. Kim, “Audiocaps: Generating captions for audios in the wild,” in NAACL-HLT, 2019. Available: https://aclanthology.org/N19-1011/

#### Clotho
[2] K. Drossos, S. Lipping, and T. Virtanen, “Clotho: An Audio Captioning Dataset,” arXiv:1910.09387 [cs, eess], Oct. 2019, Available: http://arxiv.org/abs/1910.09387

#### MACS
[3] F. Font, A. Mesaros, D. P. W. Ellis, E. Fonseca, M. Fuentes, and B. Elizalde, Proceedings of the 6th Workshop on Detection and Classication of Acoustic Scenes and Events (DCASE 2021). Barcelona, Spain: Music Technology Group - Universitat Pompeu Fabra, Nov. 2021. Available: https://doi.org/10.5281/zenodo.5770113

#### WavCaps
[1] X. Mei et al., “WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research,” arXiv preprint arXiv:2303.17395, 2023, [Online]. Available: https://arxiv.org/pdf/2303.17395.pdf 

## Cite the aac-datasets package
If you use this software, please consider cite it as below :

```
@software{
    Labbe_aac_datasets_2022,
    author = {Labbé, Etienne},
    license = {MIT},
    month = {09},
    title = {{aac-datasets}},
    url = {https://github.com/Labbeti/aac-datasets/},
    version = {0.4.0},
    year = {2023}
}
```

## Contact
Maintainer:
- Etienne Labbé "Labbeti": labbeti.pub@gmail.com
