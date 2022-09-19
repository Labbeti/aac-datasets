Usage
========================

Download a dataset
########################

You can download each dataset subset by using the download=True option in dataset constructor.

.. code-block:: python
    :caption: Download Clotho development dataset (python).

    from aac_datasets import Clotho

    _ = Clotho("/my/path/to/data", subset="dev", download=True)

You can also do the same by the command line :

.. code-block:: bash
    :caption: Download Clotho development dataset (command).

    aacd-download --root "/my/path/to/data" clotho --subsets dev


Load data
########################

Datasets are standard pytorch `Dataset`, and the can be loaded with squared brackets `[]`:

.. code-block:: python
    :caption: Load an item.

    from aac_datasets import Clotho

    clotho_dev_ds = Clotho("/my/path/to/data", subset="dev")
    print(clotho_dev_ds[0])
    # Returns the first Clotho item (audio, captions and metadata...) as dict.
    # {"audio": tensor([[...]]), "captions": ['A wild...', ...], ...}


You can also use the method `at` to get a specific data in the dataset:

.. code-block:: python
    :caption: Load only the captions.

    print(clotho_dev_ds.at(0, "captions"))
    # Returns the 5 captions of the first audio (WITHOUT loading audio):
    # ['A wild assortment of birds are chirping and calling out in nature.',
    #  'Several different types of bird are tweeting and making calls.',
    #  'Birds tweeting and chirping happily, engine in the distance.',
    #  'An assortment of  wild birds are chirping and calling out in nature.',
    #  'Birds are chirping and making loud bird noises.']


Build PyTorch DataLoader
########################

Pytorch DataLoader can be easely created for AAC datasets if you override the `collate_fn` argument.

.. code-block:: python
    :caption: Build pytorch dataloader.

    from aac_datasets import Clotho
    from aac_datasets.utils import BasicCollate

    clotho_dev_ds = Clotho("/my/path/to/data", subset="dev")
    collate = BasicCollate()

    loader = DataLoader(clotho_dev_ds, batch_size=32, collate_fn=collate)
    for batch in loader:
        ...
