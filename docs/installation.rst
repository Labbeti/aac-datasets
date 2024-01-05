Installation
============

Simply run:

.. code-block:: bash
    
    pip install aac-datasets


Python requirements
###################

The python requirements are automatically installed when using pip on this repository.

.. code-block:: bash

    torch >= 1.10.1
    torchaudio >= 0.10.1
    py7zr >= 0.17.2
    pyyaml >= 6.0
    tqdm >= 4.64.0
    huggingface-hub>=0.15.1
    numpy>=1.21.2

External requirements (AudioCaps only)
######################################

The external requirements needed to download **AudioCaps** are **ffmpeg** and **yt-dlp**.
.. These two programs can be download on Ubuntu using `sudo apt install ffmpeg youtube-dl`.
**ffmpeg** can be install on Ubuntu using `sudo apt install ffmpeg` and **yt-dlp** from the `official repo <https://github.com/yt-dlp/yt-dlp>`_.

You can also override their paths for AudioCaps:

.. code-block:: python

    from aac_datasets import AudioCaps
    dataset = AudioCaps(
        download=True,
        ffmpeg_path="/my/path/to/ffmpeg",
        ytdl_path="/my/path/to/youtube_dl",
    )
