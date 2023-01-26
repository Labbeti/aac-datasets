Usage
========================

Download a dataset
########################

You can download each dataset subset by using the download=True option in dataset constructor.
.. :caption: Download Clotho development dataset (python).

.. code-block:: python

    from aac_datasets import Clotho

    _ = Clotho("/my/path/to/data", subset="dev", download=True)

You can also do the same by using functions :
.. :caption: Download Clotho development dataset (command line).

.. code-block:: python

    from aac_datasets.download import download_clotho

    download_clotho("/my/path/to/data", subsets=("dev",), download=True)

Or by the command line :
.. :caption: Download Clotho development dataset (command line).

.. code-block:: bash

    aac-datasets-download --root "/my/path/to/data" clotho --subsets dev


Load data
########################

In this packages, the datasets are standard PyTorch `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, and each data can be loaded with squared brackets `[]`:

.. :caption: Load an item.

.. code-block:: python

    from aac_datasets import Clotho

    clotho_dev_ds = Clotho("/my/path/to/data", subset="dev")
    print(len(clotho_dev_ds))
    # 3839
    print(clotho_dev_ds[0])
    # Returns the first Clotho item (audio, captions and metadata...) as dict.
    # {"audio": tensor([[...]]), "captions": ['A wild...', ...], ...}


You can also add the column/field name that you want to load:

.. :caption: Load only the captions.

.. code-block:: python

    print(clotho_dev_ds[0, "captions"])
    # Returns the 5 captions of the first audio (but WITHOUT loading audio):
    # ['A muddled noise of broken channel of the TV',
    #  'A television blares the rhythm of a static TV.',
    #  'Loud television static dips in and out of focus',
    #  'The loud buzz of static constantly changes pitch and volume.',
    #  'heavy static and the beginnings of a signal on a transistor radio']

    print(clotho_dev_ds[10:20, "captions"])
    # Returns the captions associated to the audio of index 10 (included) to 20 (excluded)

To each the list of columns in the dataset, use the property `column_names`

.. :caption: Show the column names.

.. code-block:: python

    print(clotho_dev_ds.column_names)
    # ['audio',
    #  'captions',
    #  'dataset',
    #  'fname',
    #  'index',
    #  'subset',
    #  'sr',
    #  'keywords',
    #  'sound_id',
    #  'sound_link',
    #  'start_end_samples',
    #  'manufacturer',
    #  'license']


Build PyTorch DataLoader
########################

PyTorch DataLoader can be easely created for AAC datasets if you override the `collate_fn` argument.

.. :caption: Build PyTorch dataloader.

.. code-block:: python

    from aac_datasets import Clotho
    from aac_datasets.utils import BasicCollate

    clotho_dev_ds = Clotho("/my/path/to/data", subset="dev")
    collate = BasicCollate()

    loader = DataLoader(clotho_dev_ds, batch_size=32, collate_fn=collate)
    for batch in loader:
        # batch is a dictionary of lists, containing audio, captions, metadata...
        ...
