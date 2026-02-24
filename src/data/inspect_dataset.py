from pathlib import Path
from typing import List


class DatasetInspector:
    def __init__(self, dataset_path: str | Path):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path no encontrado: {self.dataset_path}")

        self.splits = self._detect_splits()

    def _detect_splits(self) -> List[str]:
        # Detecta qué splits existen en el dataset (train, val y/o test)
        possible_splits = ["train", "val", "test"]

        valid_splits = []

        for split in possible_splits:
            images_path = self.dataset_path / "split" / split / "images"
            labels_path = self.dataset_path / "split" / split / "labels"

            # Solo considera como splits válidos aquellos que tengan la estructura esperada de imágenes y etiquetas
            if images_path.exists() and labels_path.exists():
                valid_splits.append(split)

        return valid_splits
