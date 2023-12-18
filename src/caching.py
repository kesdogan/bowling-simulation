import pickle

from src.solver import ProjectiveDynamicsSolver
from pathlib import Path
from datetime import datetime


class Cache:
    def __init__(self, faces, vertices):
        self.positions = []
        self.faces = faces
        self.initial_vertices = vertices
        self.file_name = f"cache_{datetime.now()}.pkl"

    def add(self, positions):
        self.positions.append(positions)

    def get(self, index):
        return self.positions[min(index, len(self.positions) - 1)]

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
