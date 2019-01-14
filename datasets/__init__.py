"""Infrastructure for downloading datasets from the internet."""
from pathlib import Path

import requests


class Dataset:
    """A dataset from the internet."""

    __slots__ = "filename", "url"

    def __init__(self, filename, url):
        """Create a Dataset.

        :param filename: The filename to save the dataset to.
        :type filename: str
        :param url: The URL to download the dataset from.
        :type url: str
        """
        # Add type hints for better editor completion.
        self.filename: Path = Path(__file__).parent.joinpath(Path(filename))
        self.url: str = url

    def download(self):
        """Download this dataset from the internet.

        Overwrites the dataset if it exists on disk already.
        """
        self.curl(self.url, self.filename)

    def exists(self):
        """Determine if this dataset exists.

        :return: Whether or not this dataset exists on disk.
        :rtype: bool
        """
        return self.filename.exists()

    @staticmethod
    def curl(url, filename):
        """Download a file from the internet.

        Overwrites the file if it exists.

        :param url: The URL to Download
        :type url: str
        :param filename: The filename to save the file as
        :type filename: str
        """
        r = requests.get(url, stream=True)
        with open(filename, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)


DATASETS = {
    "nietzsche": Dataset("nietzsche.txt", "https://s3.amazonaws.com/text-datasets/nietzsche.txt")
}
