#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
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

try:
    # To support torchaudio >= 2.1.0
    from torchaudio import AudioMetaData  # type: ignore
except ImportError:
    from torchaudio.backend.common import AudioMetaData

from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeAlias, TypeGuard

from aac_datasets.utils.collections import dict_list_to_list_dict, union_dicts
from aac_datasets.utils.type_guards import (
    is_iterable_bool,
    is_iterable_int,
    is_iterable_str,
    is_list_bool,
    is_list_int,
)

pylog = logging.getLogger(__name__)

ItemType = TypeVar("ItemType", covariant=True)
IndexType: TypeAlias = Union[int, Iterable[int], Iterable[bool], Tensor, slice, None]
ColumnType: TypeAlias = Union[str, Iterable[str], None]

_INDEX_TYPES = ("int", "Iterable[int]", "Iterable[bool]", "Tensor", "slice", "None")


def _is_index(index: Any) -> TypeGuard[IndexType]:
    return (
        isinstance(index, int)
        or is_iterable_int(index)
        or is_iterable_bool(index)
        or isinstance(index, slice)
        or index is None
        or (
            isinstance(index, Tensor)
            and not index.is_floating_point()
            and not index.is_complex()
            and index.ndim in (0, 1)
        )
    )


def _is_column(column: Any) -> TypeGuard[ColumnType]:
    return is_iterable_str(column, accept_str=True) or column is None


class AACDataset(Generic[ItemType], Dataset[ItemType]):
    """Base class for AAC datasets."""

    # Initialization
    def __init__(
        self,
        raw_data: Optional[Dict[str, List[Any]]] = None,
        transform: Optional[Callable[[ItemType], Any]] = None,
        column_names: Optional[Iterable[str]] = None,
        flat_captions: bool = False,
        sr: Union[int, Iterable[int], None] = None,
        verbose: int = 0,
    ) -> None:
        if raw_data is None:
            raw_data = {}
        if column_names is None:
            column_names = raw_data.keys()
        column_names = list(column_names)
        if isinstance(sr, Iterable):
            sr = list(sr)

        if len(raw_data) > 1:
            size = len(next(iter(raw_data.values())))
            invalid_columns = [col for col, lst in raw_data.items() if len(lst) != size]
            if len(invalid_columns) > 0:
                msg = f"Invalid raw_data number of items in the following columns: {tuple(invalid_columns)}."
                raise ValueError(msg)

        super().__init__()
        self._raw_data = raw_data
        self._transform = transform
        self._columns = column_names
        self._flat_captions = flat_captions
        self._sr = sr
        self._verbose = verbose

        self._online_fns = {}
        self._sizes = []

        if self._flat_captions:
            self._flat_raw_data()

    @staticmethod
    def new_empty() -> "AACDataset":
        """Create a new empty dataset."""
        return AACDataset(
            raw_data={},
            transform=None,
            column_names=(),
            flat_captions=False,
            sr=None,
            verbose=0,
        )

    # Properties
    @property
    def all_columns(self) -> List[str]:
        """The name of all columns of the dataset."""
        return list(union_dicts(self._raw_data, self._online_fns))

    @property
    def column_names(self) -> List[str]:
        """The name of all selected column of the dataset."""
        return self._columns

    @property
    def flat_captions(self) -> bool:
        """Returns true if captions has been flattened."""
        return self._flat_captions

    @property
    def num_columns(self) -> int:
        """Number of columns in the dataset."""
        return len(self.column_names)

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset (same as len())."""
        return len(self)

    @property
    def raw_data(self) -> Dict[str, List[Any]]:
        return self._raw_data

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the dataset (number of columns, number of rows)."""
        return len(self), len(self.column_names)

    @property
    def sr(self) -> Union[int, List[int], None]:
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
        columns: Iterable[str],
    ) -> None:
        columns = list(columns)
        self._check_columns(columns)
        self._columns = columns

    @transform.setter
    def transform(self, transform: Optional[Callable[[ItemType], Any]]) -> None:
        self._transform = transform

    @verbose.setter
    def verbose(self, verbose: int) -> None:
        self._verbose = verbose

    # Public methods
    @overload
    def at(self, index: int) -> ItemType:
        ...

    @overload
    def at(  # type: ignore
        self,
        index: Union[Iterable[int], Iterable[bool], slice, None],
        column: str,
    ) -> List:
        ...

    @overload
    def at(
        self,
        index: Union[Iterable[int], Iterable[bool], slice, None],
        column: Union[Iterable[str], None] = None,
    ) -> Dict[str, List]:
        ...

    @overload
    def at(self, index: IndexType, column: ColumnType) -> Any:
        ...

    def at(
        self,
        index: IndexType = None,
        column: ColumnType = None,
    ) -> Any:
        """Get a specific data field.

        :param index: The index or slice of the value in range [0, len(dataset)-1].
        :param column: The name(s) of the column. Can be any value of :meth:`~Clotho.columns`.
        :returns: The field value. The type depends of the column.
        """
        if index is None:
            index = slice(None)

        elif isinstance(index, Tensor):
            if __debug__:
                if index.ndim not in (0, 1):
                    msg = f"Invalid number of dimensions for index argument. (found {index.ndim=} but expected 0 or 1)"
                    raise ValueError(msg)
                elif index.is_floating_point():
                    msg = "Invalid tensor dtype. (found floating-point tensor but expected integer or bool tensor)"
                    raise TypeError(msg)
                elif index.is_complex():
                    msg = "Invalid tensor dtype. (found complex tensor but expected integer or bool tensor)"
                    raise TypeError(msg)
            index = index.tolist()

        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(index, column_i) for column_i in column}

        if isinstance(index, (int, slice)) and (
            column in self._raw_data.keys() and column not in self._online_fns
        ):
            return self._raw_data[column][index]  # type: ignore

        if isinstance(index, slice):
            index = range(len(self))[index]

        if isinstance(index, Iterable):
            index = list(index)
            if is_list_bool(index):
                if len(index) != len(self):
                    msg = f"The length of the mask ({len(index)}) does not match the length of the dataset ({len(self)})."
                    raise IndexError(msg)
                index = [i for i, idx_i in enumerate(index) if idx_i]

            elif __debug__ and not is_list_int(index):
                msg = f"Invalid input type for {index=}. (expected Iterable[int], not Iterable[{index[0].__class__.__name__}])"
                raise TypeError(msg)

            values = [
                self.at(idx_i, column)
                for idx_i in tqdm.tqdm(
                    index,
                    desc=f"Loading column '{column}'...",
                    disable=self._verbose < 2,
                )
            ]
            return values

        if __debug__ and not isinstance(index, int):
            msg = (
                f"Invalid argument type {type(index)}. (expected one of {_INDEX_TYPES})"
            )
            raise TypeError(msg)

        return self._load_online_value(column, index)

    def has_raw_column(self, column: str) -> bool:
        """Returns True if column name exists in raw data."""
        return column in self._raw_data

    def has_post_column(self, column: str) -> bool:
        """Returns True if column name exists in post processed data."""
        return column in self._online_fns

    def has_column(self, column: str) -> bool:
        """Returns True if column name exists in data."""
        return self.has_raw_column(column) or self.has_post_column(column)

    def remove_column(self, column: str) -> Union[List[Any], Callable]:
        """Removes a column from this dataset."""
        if column in self._raw_data:
            column_data = self._raw_data.pop(column, [])
            return column_data
        elif column in self._online_fns:
            fn = self._online_fns.pop(column)
            return fn
        else:
            raise ValueError(f"Column '{column}' does not exists in dataset.")

    def rename_column(
        self,
        old_column: str,
        new_column: str,
        allow_replace: bool = False,
    ) -> None:
        """Renames a column from this dataset."""
        column_data_or_fn = self.remove_column(old_column)

        if isinstance(column_data_or_fn, List):
            self.add_raw_column(new_column, column_data_or_fn, allow_replace)
        elif isinstance(column_data_or_fn, Callable):
            self.add_online_column(new_column, column_data_or_fn, allow_replace)
        else:
            msg = f"Invalid type {type(column_data_or_fn)}. (expected List or Callable)"
            raise TypeError(msg)

    def add_raw_column(
        self,
        column_name: str,
        column_data: List[Any],
        allow_replace: bool = False,
    ) -> None:
        """Add a new raw column to this dataset."""
        if not allow_replace and column_name in self._raw_data:
            msg = f"Column '{column_name}' already exists. Please choose another name or set allow_replace arg to True."
            raise ValueError(msg)
        if len(self._raw_data) > 0 and len(column_data) != len(self):
            msg = f"Invalid number of rows in column '{column_name}'."
            raise ValueError(msg)
        self._raw_data[column_name] = column_data

    def add_online_column(
        self,
        column: str,
        load_fn: Callable[[Any, int], Any],
        allow_replace: bool = False,
    ) -> None:
        """Add a new post-processed column to this dataset."""
        if not allow_replace and column in self._online_fns:
            msg = f"Column '{column}' already exists in {self} and found argument {allow_replace=}."
            raise ValueError(msg)
        self._online_fns[column] = load_fn

    def add_online_columns(
        self,
        post_columns_fns: Dict[str, Callable[[Any, int], Any]],
        allow_replace: bool = False,
    ) -> None:
        """Add several new post-processed columns to this dataset."""
        for name, load_fn in post_columns_fns.items():
            self.add_online_column(name, load_fn, allow_replace)

    def preload_online_column(
        self,
        column: str,
        allow_replace: bool = False,
    ) -> Callable[[Any, int], Any]:
        """Load all data from a post-column data into raw data."""
        if column not in self._online_fns:
            msg = f"Invalid argument {column=}."
            raise ValueError(msg)

        column_data = [
            self._load_online_value(column, i)
            for i in tqdm.trange(
                len(self),
                disable=self._verbose < 2,
                desc=f"Preloading column '{column}'",
            )
        ]
        fn = self._online_fns.pop(column)
        self.add_raw_column(column, column_data, allow_replace=allow_replace)
        return fn

    def to_dict(self, load_online_values: bool = False) -> Dict[str, List[Any]]:
        """Convert dataset to dictionary.

        :param load_online_values: If True, load ALL online values (e.g. audio waveform). Otherwise load only the raw data of the dataset. defaults to False.
        """
        raw_data = copy.copy(self._raw_data)
        if load_online_values:
            for column_name in self._online_fns.keys():
                column_data = self.at(None, column_name)
                raw_data[column_name] = column_data
        return raw_data

    def to_list(self, load_online_values: bool = False) -> List[ItemType]:
        """Convert dataset to list.

        :param load_online_values: If True, load ALL online values (e.g. audio waveform). Otherwise load only the raw data of the dataset. defaults to False.
        """
        raw_data = self.to_dict(load_online_values)
        return dict_list_to_list_dict(raw_data, key_mode="same")  # type: ignore

    # Magic methods
    @overload
    def __getitem__(self, index: int) -> ItemType:
        ...

    @overload
    def __getitem__(self, index: Tuple[Union[Iterable[int], slice, None], str]) -> List:  # type: ignore
        ...

    @overload
    def __getitem__(self, index: Union[Iterable[int], slice, None]) -> Dict[str, List]:
        ...

    @overload
    def __getitem__(
        self,
        index: Tuple[Union[Iterable[int], slice, None], Union[Iterable[str], None]],
    ) -> Dict[str, List]:
        ...

    @overload
    def __getitem__(self, index: Any) -> Any:
        ...

    def __getitem__(self, index: Union[IndexType, Tuple[IndexType, ColumnType]]) -> Any:
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and _is_index(index[0])
            and _is_column(index[1])
        ):
            index, column = index
        else:
            column = None

        item = self.at(index, column)  # type: ignore

        if (
            isinstance(index, int)
            and self._transform is not None
            and (
                column is None
                or (
                    isinstance(column, Iterable)
                    and not isinstance(column, str)
                    and set(column) == set(self._columns)
                )
            )
        ):
            item = self._transform(item)  # type: ignore
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
            "num_columns": self.num_columns,
        }
        repr_str = ", ".join(f"{k}={v}" for k, v in info.items())
        return f"{self.__class__.__name__}({repr_str})"

    # Private methods
    def _check_columns(self, columns: List[str]) -> None:
        expected_columns = dict.fromkeys(self.all_columns)
        invalid_columns = [name for name in columns if name not in expected_columns]
        if len(invalid_columns) > 0:
            msg = f"Invalid argument {columns=}. (found {len(invalid_columns)} invalids column names for {self.__class__.__name__}: {invalid_columns})"
            raise ValueError(msg)

        invalid_columns = [name for name in columns if not self.has_column(name)]
        if len(invalid_columns) > 0:
            msg = f"Invalid argument {columns=}. (found {len(invalid_columns)} invalids column names for {self.__class__.__name__}: {invalid_columns})"
            raise ValueError(msg)

    def _flat_raw_data(self) -> None:
        raw_data, sizes = _flat_raw_data(self._raw_data)
        self._raw_data = raw_data
        self._sizes = sizes

    def _unflat_raw_data(self) -> None:
        raw_data = _unflat_raw_data(self._raw_data, self._sizes)
        self._raw_data = raw_data

    def _load_online_value(self, column: str, index: int) -> Any:
        if column in self._online_fns:
            fn = self._online_fns[column]
            return fn(self, index)
        else:
            raise ValueError(
                f"Invalid argument column={column} at {index=}. (expected one of {self.all_columns})"
            )

    def _load_audio(self, index: int) -> Tensor:
        fpath = self.at(index, "fpath")
        audio_and_sr: Tuple[Tensor, int] = torchaudio.load(fpath)  # type: ignore
        audio, sr = audio_and_sr

        if not __debug__:
            return audio

        # Sanity check
        if audio.nelement() == 0:
            msg = f"Invalid audio number of elements in {fpath}. (expected {audio.nelement()=} > 0)"
            raise RuntimeError(msg)

        if self._sr is not None and (self._sr != sr):
            msg = (
                f"Invalid sample rate {sr}Hz for audio {fpath}. (expected {self._sr}Hz)"
            )
            raise RuntimeError(msg)
        return audio

    def _load_audio_metadata(self, index: int) -> AudioMetaData:
        fpath = self.at(index, "fpath")
        audio_metadata = torchaudio.info(fpath)  # type: ignore
        return audio_metadata

    def _load_duration(self, index: int) -> float:
        audio_metadata: AudioMetaData = self.at(index, "audio_metadata")
        duration = audio_metadata.num_frames / audio_metadata.sample_rate
        return duration

    def _load_fname(self, index: int) -> str:
        fpath = self.at(index, "fpath")
        fname = osp.basename(fpath)
        return fname

    def _load_num_channels(self, index: int) -> int:
        audio_metadata = self.at(index, "audio_metadata")
        num_channels = audio_metadata.num_channels
        return num_channels

    def _load_num_frames(self, index: int) -> int:
        audio_metadata = self.at(index, "audio_metadata")
        num_frames = audio_metadata.num_frames
        return num_frames

    def _load_sr(self, index: int) -> int:
        audio_metadata = self.at(index, "audio_metadata")
        sr = audio_metadata.sample_rate
        return sr


def _flat_raw_data(
    raw_data: Dict[str, List[Any]],
    caps_column: str = "captions",
) -> Tuple[Dict[str, List[Any]], List[int]]:
    if caps_column not in raw_data:
        msg = f"Cannot flat raw data without '{caps_column}' column. (found only columns {tuple(raw_data.keys())})"
        raise ValueError(msg)

    mcaps: List[List[str]] = raw_data[caps_column]
    raw_data_flat = {key: [] for key in raw_data.keys()}

    for i, caps in enumerate(mcaps):
        if len(caps) == 0:
            for key in raw_data.keys():
                raw_data_flat[key].append(raw_data[key][i])
        else:
            for cap in caps:
                for key in raw_data.keys():
                    if key == caps_column:
                        continue
                    raw_data_flat[key].append(raw_data[key][i])

                # Overwrite cap
                raw_data_flat[caps_column].append([cap])

    sizes = [len(caps) for caps in mcaps]
    return raw_data_flat, sizes


def _unflat_raw_data(
    raw_data_flat: Dict[str, List[Any]],
    sizes: List[int],
    caps_column: str = "captions",
) -> Dict[str, List[Any]]:
    if caps_column not in raw_data_flat:
        msg = f"Cannot flat raw data without '{caps_column}' column. (found only columns {tuple(raw_data_flat.keys())})"
        raise ValueError(msg)

    raw_data = {key: [] for key in raw_data_flat.keys()}

    cumsize = 0
    for size in sizes:
        for key in raw_data.keys():
            if key == caps_column:
                caps = [
                    raw_data_flat[key][index][0]
                    for index in range(cumsize, cumsize + size)
                ]
                raw_data[key].append(caps)
            else:
                raw_data[key].append(raw_data_flat[key][cumsize])
        cumsize += size

    return raw_data
