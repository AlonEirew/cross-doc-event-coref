from os import path
from pathlib import Path

LIBRARY_PATH = Path(path.realpath(__file__)).parent
LIBRARY_ROOT = LIBRARY_PATH.parent
