import pickle
from datetime import datetime
from pathlib import Path

from src.solver import ProjectiveDynamicsSolver


class Cache:
    """Allows to store a simulation in a file and load it later."""

    def __init__(self, faces, vertices):
        self.positions = []
        self.faces = faces
        self.initial_vertices = vertices
        self.file_name = f"cache/cache_{datetime.now()}.pkl"

    def add_frame(self, positions):
        self.positions.append(positions)

    def get_frame(self, index):
        return self.positions[min(index, len(self.positions) - 1)]

    def store(self):
        with open(self.file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, file_name=None) -> "Cache":
        """Loads the last cached file. If file_name is given, loads that file instead."""
        last_cached_file = file_name or max(
            Path().glob("cache/cache*.pkl"), key=lambda f: f.stat().st_mtime
        )
        with open(last_cached_file, "rb") as f:
            return pickle.load(f)
