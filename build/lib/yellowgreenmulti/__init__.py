"""

"""
import logging

from .version import __version__


__all__ = ['__version__']


# Add a NullHandler to the library root logger as it is up to the user to configure logging
logging.getLogger(__package__).addHandler(logging.NullHandler())
