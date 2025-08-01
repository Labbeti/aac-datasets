About datasets subsets
========================

AudioCaps
########################
The original AudioCaps (V1) dataset contains only 3 subsets : `train`, `val` and `test`.

A fourth subset for V1, named `train_fixed` is another version of the `train` subset where captions has been manually corrected or deleted. I recommend to use it over `train` to train models.
For more details, see paper `"CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding" <https://arxiv.org/abs/2309.00454>`_.

In 2025, a new AudioCaps version has been released and contains more data, with the subsets: `train`, `val` and `test`. `train_fixed` is not available for that version.

Clotho
########################
Clotho v2.1 contains 7 subsets:

- `dev` : contains 3.8K files for training,
- `val` : contains 1K files for validation,
- `eval` : contains 1K files for testing,
- `dcase_aac_test` : contains 1K files without captions used in the evaluation of the DCASE challenge audio captioning task (AAC),
- `dcase_aac_analysis` : contains 6K audio files without captions used in the evaluation of the DCASE challenge audio captioning task (AAC),
- `dcase_t2a_audio` : contains 1K audio files without captions used in the evaluation of the DCASE challenge language based audio retrieval task (also called Text-to-Audio retrieval, T2A),
- `dcase_t2a_captions` : contains 1K captions (queries) without audios files used in the evaluation of the DCASE challenge language based audio retrieval task (also called Text-to-Audio retrieval, T2A).

Older Clotho versions (v1 and v2) contains less data or minor typo errors and are not recommanded.

In the DCASE challenge for Audio Captioning, organizers followed a `different convention <https://dcase.community/challenge2022/task-automatic-audio-captioning#development-validation-and-evaluation-datasets-of-clotho>`_ for the subsets names.

.. list-table:: Clotho subsets names
   :header-rows: 1

   * - Clotho convention
     - DCASE convention
   * - dev
     - development-training
   * - val
     - development-validation
   * - eval
     - development-testing
   * - dcase_aac_test
     - evaluation (-testing)
   * - dcase_aac_analysis
     - analysis

MACS
########################
MACS contains only 1 subset: `full`. Its data is typically used as additional training data.

WavCaps
########################
WavCaps contains 7 subsets. **Recommanded subsets for training are**:

- `bbc` : contains 31K files from BBC Sound Effects website,
- `soundbible` : contains 1.2K files from SoundBible website,
- `audioset_no_audiocaps_v1` : contains 99K files from `audioset` subset without overlapping data with AudioCaps,
- `freesound_no_clotho_v2` : contains 257K files from `freesound` subset without overlapping data with Clotho.

Other subsets exists but contains overlaps with several test datasets, and are therefore not recommanded:

- `audioset` : contains 108K files from AudioSet strongly labeled dataset,
- `freesound` : contains 262K files from FreeSound website.

Since WavCaps does not contains validation or testing subsets, all of their data is used as additional training data.
The subsets `audioset_no_audiocaps_v1` and `freesound_no_clotho_v2` are provided to avoid biases when evaluating on AudioCaps or Clotho datasets (except for Clotho `dcase_aac_test`).

The subset `freesound_no_clotho` was removed in aac-datasets 0.6.0. It was containing 258K files from `freesound` subset without overlapping data with Clotho (except for subsets of DCASE2023 task 6a).

Datasets overlaps
########################
Audio-Text datasets typically comme from other audio classification datasets or similar websites, which might lead to overlaps that can create data leaks in your training.
Here is a list of known overlaps between differents sound events that should be aware of:

.. list-table:: Overlaps between sound event datasets
   :header-rows: 1

   * - Dataset A
     - Dataset B
     - Proportion of A in B (%)
   * - AudioCaps
     - AudioSet (train)
     - 100
   * - Clotho
     - FSD50K (train)
     - 5.4
   * - AudioCaps
     - WavCaps (audioset)
     - 17.6
   * - Clotho
     - WavCaps (freesound)
     - 89.0

If you do not take this overlaps into account, you might overestimate your results of your AAC model.
Overlaps with WavCaps can be avoided by using `audioset_no_audiocaps_v1` and `freesound_no_clotho_v2` subsets.
