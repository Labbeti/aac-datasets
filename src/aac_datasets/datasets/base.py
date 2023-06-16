#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torchaudio
import tqdm

from torchaudio.backend.common import AudioMetaData
from typing_extensions import TypedDict

from torch import Tensor
from torch.utils.data.dataset import Dataset

from aac_datasets.utils.collate import intersect_lists


pylog = logging.getLogger(__name__)


class DatasetCard:
    pass


ItemType = TypeVar("ItemType", bound=TypedDict, covariant=True)


class AACDataset(Generic[ItemType], Dataset[ItemType]):
    """Base class for AAC datasets."""

    _AUTO_COLUMNS = {
        "audio": ["fpath"],
        "audio_metadata": ["fpath"],
        "duration": ["audio_metadata"],
        "fname": ["fpath"],
        "num_channels": ["audio_metadata"],
        "num_frames": ["audio_metadata"],
        "sr": ["audio_metadata"],
    }

    # Initialization
    def __init__(
        self,
        raw_data: Dict[str, List[Any]],
        transform: Optional[Callable],
        column_names: Iterable[str],
        flat_captions: bool,
        sr: Optional[int],
        verbose: int,
    ) -> None:
        column_names = list(column_names)

        if len(raw_data) > 1:
            size = len(next(iter(raw_data.values())))
            invalid_columns = [col for col, lst in raw_data.items() if len(lst) != size]
            if len(invalid_columns) > 0:
                raise ValueError(
                    f"Invalid raw_data number of items in the following columns: {tuple(invalid_columns)}."
                )

        super().__init__()
        self._raw_data = raw_data
        self._transform = transform
        self._column_names = column_names
        self._flat_captions = flat_captions
        self._sr = sr
        self._verbose = verbose
        self._auto_columns_fns = AACDataset.default_auto_columns_fns()

        if self._flat_captions:
            self._flat_raw_data()

        # Check must be done after setting the attributes
        self._check_column_names(column_names)

    @staticmethod
    def new_empty() -> "AACDataset":
        return AACDataset({}, None, (), False, None, 0)

    @classmethod
    def default_auto_columns_fns(cls) -> dict[str, Callable]:
        return {
            "audio": AACDataset._load_audio,
            "audio_metadata": AACDataset._load_audio_metadata,
            "duration": AACDataset._load_duration,
            "fname": AACDataset._load_fname,
            "num_channels": AACDataset._load_num_channels,
            "num_frames": AACDataset._load_num_frames,
            "sr": AACDataset._load_sr,
        }

    # Properties
    @property
    def all_column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        return list(
            self._raw_data
            | dict.fromkeys(filter(self._can_be_loaded, self._AUTO_COLUMNS))
        )

    @property
    def column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        return self._column_names

    @property
    def flat_captions(self) -> bool:
        """The name of each column of the dataset."""
        return self._flat_captions

    @property
    def raw_data(self) -> Dict[str, List[Any]]:
        return self._raw_data

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the dataset."""
        return len(self), len(self.column_names)

    @property
    def sr(self) -> Optional[int]:
        return self._sr

    @property
    def transform(self) -> Optional[Callable]:
        return self._transform

    @property
    def verbose(self) -> int:
        return self._verbose

    @column_names.setter
    def column_names(
        self,
        column_names: Iterable[str],
    ) -> None:
        column_names = list(column_names)
        self._check_column_names(column_names)
        self._column_names = column_names

    @transform.setter
    def transform(self, transform: Optional[Callable]) -> None:
        self._transform = transform

    # Public methods
    @overload
    def at(self, idx: int) -> ItemType:
        ...

    @overload
    def at(self, idx: Union[Iterable[int], slice, None], column: str) -> List:
        ...

    @overload
    def at(
        self, idx: Union[Iterable[int], slice, None], column: Union[Iterable[str], None]
    ) -> Dict[str, List]:
        ...

    @overload
    def at(self, idx: Any, column: Any) -> Any:
        ...

    def at(
        self,
        idx: Union[int, Iterable[int], None, slice] = None,
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        """Get a specific data field.

        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :param column: The name(s) of the column. Can be any value of :meth:`~Clotho.column_names`.
        :returns: The field value. The type depends of the column.
        """
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, Tensor):
            if idx.ndim not in (0, 1):
                raise ValueError(
                    f"Invalid number of dimensions for idx argument. (found idx.ndim={idx.ndim} but expected 0 or 1)"
                )
            elif idx.is_floating_point():
                raise TypeError(
                    f"Invalid tensor dtype. (found floating-point tensor but expected integer tensor)"
                )
            idx = idx.tolist()

        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        if isinstance(idx, (int, slice)) and column in self._raw_data.keys():
            return self._raw_data[column][idx]  # type: ignore

        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if isinstance(idx, Iterable):
            idx = list(idx)
            if all(isinstance(idx_i, bool) for idx_i in idx):
                if len(idx) != len(self):
                    raise IndexError(
                        f"The length of the mask ({len(idx)}) does not match the length of the dataset ({len(self)})."
                    )
                idx = [i for i, idx_i in enumerate(idx) if idx_i]

            elif not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(
                    f"Invalid input type for idx={idx}. (expected Iterable[int], not Iterable[{idx.__class__.__name__}])"
                )

            values = [
                self.at(idx_i, column)
                for idx_i in tqdm.tqdm(idx, disable=self._verbose < 2)
            ]
            return values

        if isinstance(idx, int):
            return self._load_auto_value(column, idx)
        else:
            raise TypeError(f"Invalid argument type {type(idx)}.")

    def is_loaded_column(self, name: str) -> bool:
        return name in self._raw_data

    def add_column(
        self,
        column_name: str,
        column: List[Any],
        allow_replace: bool = False,
    ) -> None:
        if not allow_replace and column_name in self._raw_data:
            raise ValueError(
                f"Column '{column_name}' already exists. Please choose another name or set allow_replace arg to True."
            )
        if len(self._raw_data) > 0 and len(column) != len(self):
            raise ValueError(f"Invalid number of rows in column '{column_name}'.")
        self._raw_data[column_name] = column

    def remove_column(self, column_name: str) -> List[Any]:
        if column_name not in self._raw_data:
            raise ValueError(f"Column '{column_name}' does not exists in data.")
        column = self._raw_data.pop(column_name, [])
        return column

    def rename_column(
        self,
        old_name: str,
        new_name: str,
        allow_replace: bool = False,
    ) -> None:
        column = self.remove_column(old_name)
        self.add_column(new_name, column, allow_replace)

    def register_auto_column(
        self,
        column_name: str,
        load_fn: Callable[["AACDataset", int], Any],
    ) -> None:
        if column_name in self.column_names:
            raise ValueError(f"Column '{column_name}' already exists in {self}.")
        self._auto_columns_fns[column_name] = load_fn

    # Magic methods
    @overload
    def __getitem__(self, idx: int) -> ItemType:
        ...

    @overload
    def __getitem__(self, idx: tuple[Union[Iterable[int], slice, None], str]) -> List:
        ...

    @overload
    def __getitem__(self, idx: Union[Iterable[int], slice, None]) -> Dict[str, List]:
        ...

    @overload
    def __getitem__(
        self, idx: tuple[Union[Iterable[int], slice, None], Union[Iterable[str], None]]
    ) -> Dict[str, List]:
        ...

    @overload
    def __getitem__(self, idx: Any) -> Any:
        ...

    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None

        item = self.at(idx, column)
        if (
            isinstance(idx, int)
            and (column is None or column == self._column_names)
            and self._transform is not None
        ):
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        """
        :return: The number of items in the dataset.
        """
        if len(self._raw_data) > 0:
            return len(next(iter(self._raw_data.values())))
        else:
            return 0

    def __repr__(self) -> str:
        info = {
            "size": len(self),
            "num_columns": len(self.column_names),
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in info.items())
        return f"{self.__class__.__name__}({repr_str})"

    # Private methods
    def _check_column_names(self, column_names: List[str]) -> None:
        expected_column_names = dict.fromkeys(self.all_column_names)
        invalid_column_names = [
            name for name in column_names if name not in expected_column_names
        ]
        if len(invalid_column_names) > 0:
            raise ValueError(
                f"Invalid argument column_names={column_names}. (found {len(invalid_column_names)} invalids column names)"
            )

        invalid_column_names = [
            name for name in column_names if not self._can_be_loaded(name)
        ]
        if len(invalid_column_names) > 0:
            raise ValueError(
                f"Invalid argument column_names={column_names}. (found {len(invalid_column_names)} invalids column names)"
            )

    def _can_be_loaded(self, col_name: str) -> bool:
        AUTO_COLUMNS = self.__class__._AUTO_COLUMNS
        if self.is_loaded_column(col_name):
            return True
        elif col_name in AUTO_COLUMNS:
            requirements = AUTO_COLUMNS[col_name]
            return all(self._can_be_loaded(req) for req in requirements)
        else:
            return False

    def _flat_raw_data(self) -> None:
        self._raw_data = _flat_raw_data(self._raw_data)

    def _load_auto_value(self, column: str, idx: int) -> Any:
        if column in self._auto_columns_fns:
            fn = self._auto_columns_fns[column]
            return fn(self, idx)
        else:
            raise ValueError(
                f"Invalid argument column={column} at idx={idx}. (expected one of {self.all_column_names})"
            )

    def _load_audio(self, idx: int) -> Tensor:
        fpath = self.at(idx, "fpath")
        audio, sr = torchaudio.load(fpath)  # type: ignore

        # Sanity check
        if audio.nelement() == 0:
            raise RuntimeError(
                f"Invalid audio number of elements in {fpath}. (expected audio.nelement()={audio.nelement()} > 0)"
            )

        if self._sr is not None and (self._sr != sr):
            raise RuntimeError(
                f"Invalid sample rate {sr}Hz for audio {fpath}. (expected {self._sr}Hz)"
            )
        return audio

    def _load_audio_metadata(self, idx: int) -> AudioMetaData:
        fpath = self.at(idx, "fpath")
        audio_metadata = torchaudio.info(fpath)  # type: ignore
        return audio_metadata

    def _load_duration(self, idx: int) -> float:
        audio_metadata = self.at(idx, "audio_metadata")
        duration = audio_metadata.num_frames / audio_metadata.sample_rate
        return duration

    def _load_fname(self, idx: int) -> str:
        fpath = self.at(idx, "fpath")
        fname = osp.basename(fpath)
        return fname

    def _load_num_channels(self, idx: int) -> int:
        audio_metadata = self.at(idx, "audio_metadata")
        num_channels = audio_metadata.num_channels
        return num_channels

    def _load_num_frames(self, idx: int) -> int:
        audio_metadata = self.at(idx, "audio_metadata")
        num_frames = audio_metadata.num_frames
        return num_frames

    def _load_sr(self, idx: int) -> int:
        audio_metadata = self.at(idx, "audio_metadata")
        sr = audio_metadata.sample_rate
        return sr


def _flat_raw_data(
    raw_data: Dict[str, List[Any]],
    caps_column_name: str = "captions",
) -> Dict[str, List[Any]]:
    if caps_column_name not in raw_data:
        raise ValueError(f"Cannot flat raw data without '{caps_column_name}' column.")

    raw_data_flat = {key: [] for key in raw_data.keys()}
    mcaps = raw_data[caps_column_name]

    for i, caps in enumerate(mcaps):
        if len(caps) == 0:
            for key in raw_data.keys():
                raw_data_flat[key].append(raw_data[key][i])
        else:
            for cap in caps:
                for key in raw_data.keys():
                    raw_data_flat[key].append(raw_data[key][i])
                raw_data_flat[caps_column_name] = [cap]

    return raw_data_flat
