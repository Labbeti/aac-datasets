# Change log

All notable changes to this project will be documented in this file.

## [0.6.0] UNRELEASED
### Added
- Methods `to_dict` and `to_list` to datasets classes.
- AudioCaps support for version `v2`.

### Changed
- Rename AudioCaps v1 `train_v2` subset to `train_fixed` to avoid confusion with AudioCaps v2 `train` subset.
- Rename WavCaps `audioset_no_audiocaps` subset to `audioset_no_audiocaps_v1` to specify which AudioCaps version is excluded.

### Fixed
- Remove invalid warning when using WavCaps subset `freesound_no_clotho_v2`.
- Download link for AudioCaps V1 subset `train_fixed`.

### Removed
- Remove subset `freesound_no_clotho` for WavCaps since it is confusing with `freesound_no_clotho_v2` and should not be used.

## [0.5.2] 2024-03-23
### Added
- `freesound_no_clotho_v2` subset to WavCaps to avoid all bias with Clotho test and analysis subsets.

## [0.5.1] 2024-03-04
### Fixed
- WavCaps download preparation (#3).
- `safe_rmdir` function when sub-directories are deleted.

## [0.5.0] 2024-01-05
### Changed
- Update typing for paths with python class `Path`.
- Refactor functional interface to load raw metadata for each dataset.
- Refactor class variables to init arguments.
- Faster AudioCaps download with `ThreadPoolExecutor`.

## [0.4.1] 2023-10-25
### Added
- `AudioCaps.DOWNLOAD_AUDIO` class variable for compatibility with [audiocaps-download 1.0](https://github.com/MorenoLaQuatra/audiocaps-download).

### Changed
- Set log level to WARNING if verbose<=0 in check.py and download.py scripts.
- Use `yt-dlp` instead of `youtube-dl` as backend to download AudioCaps audio files.. ([#1](https://github.com/Labbeti/aac-datasets/issues/1))
- Update default download message for AudioCaps. ([#1](https://github.com/Labbeti/aac-datasets/issues/1))
- Update error message when checksum is invalid for Clotho and MACS datasets. ([#2](https://github.com/Labbeti/aac-datasets/issues/2))

## [0.4.0] 2023-09-25
### Added
- First experimental implementation of **WavCaps** dataset.
- Subsets `dcase_t2a_audio` and `dcase_t2a_captions` from the DCASE Challenge task 6b, in Clotho dataset.
- Subset `train_v2` for AudioCaps dataset.
- Dataset cards as separate dataclasses for each dataset.
- Get and set global user paths for root, ffmpeg and ytdl.
- Base class for all datasets to simplify manipulation of loaded data.

### Changed
- Rename `test` subset to `dcase_aac_test`, `analysis` subset to `dcase_aac_analysis` from the DCASE Challenge task 6a, in Clotho dataset.
- Function `get_install_info` now returns `package_path`.

## [0.3.3] 2023-05-11
### Added
- Script check.py now check if the audio files exists.
- Option `VERIFY_FILES` for Clotho and MACS datasets to validate checksums.
- `CITATION` global constant for each dataset.

### Changed
- Methods `at` and `getitem` now use correct typing when passing an integer, list, slice or None values.

### Fixed
- Python minimal version in README and pyproject.toml.
- Transform applied in `getitem` method when argument is not an integer.
- Incompatibility with `torchaudio>=2.0`.
- Remove 'tags' from AudioCaps columns when with_tags=False.

## [0.3.2] 2023-01-30
### Added
- `AudioCaps.load_class_labels_indices` to load AudioSet classes map externally.
- Compatibility and tests from Python 3.7 to 3.10.

### Changed
- Attributes in datasets classes are now weakly private.
- Documentation theme and descriptions.

### Fixed
- Workflow badge with Github changes. (https://github.com/badges/shields/issues/8671)

## [0.3.1] 2022-10-31
### Changed
- AudioCaps, Clotho and MACS order are now defined by their order in the corresponding captions CSV files when available.
- Update documentation usage and main page.

### Fixed
- Workflow when requirements cache is invalid.

## [0.3.0] 2022-09-28
### Added
- Add `column_names`, `info` and `shape` properties in datasets.
- Add `is_loaded` and `set_transform` methods in datasets.
- Add column argument for method `getitem` in datasets.
- Entrypoints for command line scripts `aac-datasets-check`, `aac-datasets-download` and `aac-datasets-info`.

### Changed
- Enforce datasets order to sort by filename to avoid different orders returned by `os.listdir`.
- Function `check_directory` now returns the length of each dataset found in directory.
- Rename `get_field` methods in datasets by `at` and add support for Iterable of keys and None key.
- Change `at` arguments order and names.
- Split `BasicCollate` into 2 classes: `BasicCollate` without padding and `AdvancedCollate` with padding options.
- Weak private methods are now strongly private in datasets.
- Rename `item_transform` to `transform` in datasets.
- Rename `load_tags` to `with_tags` in `AudioCaps`.

### Fixed
- AudioCaps loading when `with_tags` is False.
- Clotho files download.

## [0.2.0] 2022-08-30
### Added
- CHANGELOG file.
- First version of the API documentation.
- Supports slicing and list indexing for the three datasets.
- Competence values for MACS annotators.
- Fields scene_label and identifier from TAU Urban acoustic scene dataset in MACS.
- Add `examples/dataloader.ipynb` notebook.

### Changed
- Update README with PyPI install and software citation.
- Download functions returns the datasets downloaded.
- MACS now have a subset parameter.
- Underscores in functions names to avoid import private functions.
- Function `aac_datasets.check.check_directory` now returns only the list of subsets loaded.
- Replace function `torchaudio.datasets.utils.download_url` by `torch.hub.download_url_to_file` to keep compatibility with future torchaudio version v0.12.
- Rename `get_raw` methods in datasets by `get_field` and add support for slicing and multi-indexing.

### Fixed
- LICENCE.txt and MACS_competence.yaml download for MACS dataset.
- Clotho download archives files.

### Removed
- Transforms dictionary in datasets.
- Argument item_type in datasets.
- Method `get` in datasets.

## [0.1.1] 2022-06-10
### Added
- CITATION file.

### Changed
- MACS now downloads only the required TAU Urban Sound archive files.
- Documentation for arguments in dataset constructors.

### Fixed
- Clotho analysis subset download and preparation.

## [0.1.0] 2022-06-07
### Added
- Initial versions of Clotho, AudioCaps and MACS pytorch dataset code.
- Download and check scripts.
