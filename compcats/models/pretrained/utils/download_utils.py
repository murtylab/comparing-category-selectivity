import os
import torch

def download_weights(
    url, filename, cache_dir=None
):
    """
    Downloads the weights
    """
    # Error handling
    file_extension = filename.split('.')[-1]
    possible_extensions = ['pth', 'pt']
    if file_extension not in possible_extensions:
        raise ValueError(
            f"Invalid file extension: {file_extension}. Possible extensions: {possible_extensions}"
        )

    # Directory to save the weights
    hf_home_dir = os.getenv('HF_HOME')
    if cache_dir is None:
        if hf_home_dir:
            cache_dir = f'{hf_home_dir}/cache'
        else:
            cache_dir = './cache'

    def check(path):
        if not os.path.exists(os.path.join(path, filename)):
            return False
        return True

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if check(cache_dir):
        pass
        # print(f"Using cached {cache_dir}")
    else:
        print(f"Downloading checkpoint to {cache_dir}")
        torch.hub.download_url_to_file(
            url=url, dst=os.path.join(cache_dir, filename)
        )
    return os.path.join(cache_dir, filename)
