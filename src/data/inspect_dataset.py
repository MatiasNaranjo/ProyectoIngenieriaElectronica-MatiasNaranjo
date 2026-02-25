from pathlib import Path
from typing import Dict, List


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

    def count_images_per_split(self) -> Dict[str, int]:
        # Cuenta cuántas imágenes hay en cada split (train, val, test)

        image_counts = {}

        for split in self.splits:
            # Para cada split
            images_dir = self.dataset_path / "split" / split / "images"

            # Crea una lista de archivos que son imágenes
            images = [
                f
                for f in images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
            # Guarda el conteo de imágenes para ese split
            image_counts[split] = len(images)

        return image_counts

    def compute_split_percentages(self) -> Dict[str, float]:
        # Calcula el porcentaje de imágenes que representa cada split respecto al total

        # Primero obtiene el conteo de imágenes por split
        counts = self.count_images_per_split()

        # Luego suma el total de imágenes en todos los splits
        total = sum(counts.values())

        if total == 0:
            raise ValueError("El dataset no contiene imágenes.")

        # Calcula el porcentaje para cada split y lo retorna
        return {
            split: round((count / total) * 100, 2) for split, count in counts.items()
        }

    def summary_split(self) -> None:
        counts = self.count_images_per_split()
        percentages = self.compute_split_percentages()
