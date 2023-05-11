# Change log

All notable changes to this project will be documented in this file.

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
