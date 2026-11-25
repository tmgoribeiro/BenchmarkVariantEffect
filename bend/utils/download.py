"""
download.py
===========
Utility functions for downloading pretrained models from the web.
"""
import os
import urllib.request


def download_model_zenodo(base_url: str, destination_dir: str = './pretrained_models'):
    """
    Download a HF model hosted as a Zenodo record.
    We use this to get the GROVER model, but it should work for any model hosted on Zenodo as a flat directory.

    Parameters
    ----------
    base_url : str
        Base URL to download from.
    destination_dir : str
        Destination directory to download to.
        Default is ./pretrained_models/
    """
    os.makedirs(destination_dir, exist_ok=True)
    files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
    for file in files:
        url = f'{base_url}/files/{file}'
        print(f'Downloading {url}')
        urllib.request.urlretrieve(url, os.path.join(destination_dir, file))
