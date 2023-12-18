import pickle

from src.solver import ProjectiveDynamicsSolver
from pathlib import Path


class Cache:
    def __init__(self, faces, vertices):
        self.cache = []
        self.faces = faces
        self.vertices = vertices

        num_existing_cached_files = len(list(Path().glob("cache*.pkl")))
        self.file_name = f"cache_{num_existing_cached_files}.pkl"

    def add(self, positions):
        self.cache.append(positions)

    def get(self, index):
        return self.cache[min(index, len(self.cache) - 1)]

    def store(self):
        with open(self.file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_name=None):
        last_cached_file = file_name or max(
            Path().glob("cache*.pkl"), key=lambda f: f.stat().st_mtime
        )
        with open(last_cached_file, "rb") as f:
            return pickle.load(f)


def play_from_cache():
    cache = Cache()
    cache.load()
