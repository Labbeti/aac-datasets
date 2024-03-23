About datasets subsets
========================

AudioCaps
########################
The original AudioCaps dataset contains only 3 subsets : `train`, `val` and `test`.

A fourth subset named `train_v2` is another version of the train subset where captions has been manually corrected or deleted. For more details, see paper `"CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding" <https://arxiv.org/abs/2309.00454>`_.

Clotho
########################
Clotho contains 7 subsets:

- `dev` : contains 3.8K files for training,
- `val` : contains 1K files for validation,
- `eval` : contains 1K files for testing,
- `dcase_aac_test` : contains 1K files without captions used in the DCASE challenge task 6a (AAC),
- `dcase_aac_analysis` : contains 6K audio files without captions used in the DCASE challenge task 6a (AAC),
- `dcase_t2a_audio` : contains 1K audio files without captions used in the DCASE challenge task 6b (Text-to-Audio retrieval),
- `dcase_t2a_captions` : contains 1K captions (queries) without audios files used in the DCASE challenge task 6b (Text-to-Audio retrieval).

In the DCASE challenge for Audio Captioning, organizers followed a `different convention <https://dcase.community/challenge2022/task-automatic-audio-captioning#development-validation-and-evaluation-datasets-of-clotho>`_ about the subsets names.

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
WavCaps contains 6 subsets:

- `audioset` : contains 108K files from AudioSet strongly labeled dataset,
- `bbc` : contains 31K files from BBC Sound Effects website,
- `freesound` : contains 262K files from FreeSound website,
- `soundbible` : contains 1.2K files from SoundBible website,
- `audioset_no_audiocaps` : contains 99K files from as subset without overlapping data with AudioCaps,
- `freesound_no_clotho` : contains 258K files from fsd subset without overlapping data with Clotho (except for subsets of task 6a).

Since WavCaps does not contains validation or testing subsets, all of their data is used as additional training data.
The subsets `audioset_no_audiocaps` and `freesound_no_clotho` are provided to avoid biases when evaluating on AudioCaps or Clotho datasets (except for Clotho `dcase_aac_test`).

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
