About datasets subsets
========================

AudioCaps
########################
The original AudioCaps dataset contains only 3 subsets : `train`, `val` and `test`.

A fourth subset named `train_v2` is another version of the train subset where captions has been manually corrected or deleted. For more details, see paper "CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding".

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


MACS
########################
MACS contains only 1 subset: `full`. Its data is typically used as additional training data.

WavCaps
########################
WavCaps contains 6 subsets:

- `as` : contains 108K files from AudioSet strongly labeled dataset,
- `bbc` : contains 31K files from BBC Sound Effects website,
- `fsd` : contains 262K files from FreeSound website,
- `sb` : contains 1.2K files from SoundBible website,
- `as_noac` : contains 99K files from as subset without overlapping data with AudioCaps,
- `fsd_nocl` : contains 258K files from fsd subset without overlapping data with Clotho (except for subsets of task 6a).

Since WavCaps does not contains validation or testing subsets, all of their data is used as additional training data.
The subsets as_noac and `fsd_nocl` are provided to avoid biases when evaluating on AudioCaps or Clotho datasets.
