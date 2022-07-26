# Change log

All notable changes to this project will be documented in this file.

## [1.0.0] (unreleased_date_TODO)
### Added
- CHANGELOG file.
- First version of the API documentation.
- Supports slicing and list indexing for the three datasets.
- Competence values for MACS annotators.
- Fields scene_label and identifier from TAU Urban acoustic scene dataset in MACS.

### Changed
- Update README with PyPI install and software citation.
- Download functions returns the datasets downloaded.
- MACS now have a subset parameter.
- Underscores in functions names to avoid import private functions.
- Function `aac_datasets.check.check_directory` now returns only the list of subsets loaded.
- Replace function `torchaudio.datasets.utils.download_url` by `torch.hub.download_url_to_file` to keep compatibility with future torchaudio version v0.12.

### Fixed
- LICENCE.txt and MACS_competence.yaml download for MACS dataset.

### Removed
- Transforms dictionary in datasets.
- Argument item_type in datasets.

## [0.1.1] 2022-06-10
### Added
- CITATION file

### Changed
- MACS now downloads only the required TAU Urban Sound archive files
- Documentation for arguments in dataset constructors

### Fixed
- Clotho analysis subset download and preparation

## [0.1.0] 2022-06-07
### Added
- Initial versions of Clotho, AudioCaps and MACS pytorch dataset code
- Download & check scripts
