from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml


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

    def total_labels_per_class(self) -> dict[str, int]:
        # Retorna el total de labels por clase sumando todos los splits
        # usando la misma lógica que _label_counts_from_files.
        return self._label_counts_from_files()

    def _load_class_names(self) -> dict[int, str]:
        """Carga el mapping id -> nombre de clase desde `dataset_meta.yaml` si existe.

        Espera que el YAML tenga una clave `names` como:
        - lista: ["cls0", "cls1", ...]
        - dict: {0: "cls0", 1: "cls1", ...}
        """

        # `dataset_meta.yaml` se copia al root del dataset durante el merge.
        meta_path = self.dataset_path / "dataset_meta.yaml"
        if not meta_path.exists():
            return {}

        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        names = data.get("names")

        if isinstance(names, dict):
            # Formato dict (a veces viene con keys como strings en YAML).
            return {int(k): str(v) for k, v in names.items()}

        if isinstance(names, list):
            # Formato lista: el índice es el id de clase.
            return {idx: str(name) for idx, name in enumerate(names)}

        return {}

    def _label_counts_from_files(self) -> dict[str, int]:
        """Cuenta labels por clase leyendo los `.txt` en formato YOLO.

        Supuesto YOLO: cada línea comienza con `class_id` (entero), seguido de coordenadas.
        Se suman todas las instancias anotadas (no imágenes).
        """

        # `class_id -> nombre`. Si no hay YAML o falta una clase, se usa el id como string.
        names_map = self._load_class_names()
        counter: dict[str, int] = defaultdict[str, int](int)

        for split in self.splits:
            split_dir = self.dataset_path / "split" / split / "labels"
            if not split_dir.exists():
                continue

            for label_file in split_dir.glob("*.txt"):
                with label_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # YOLO clásico: "class x_center y_center width height"
                        parts = line.split()
                        try:
                            class_id = int(parts[0])
                        except (ValueError, IndexError):
                            # Línea inválida, se ignora
                            continue

                        class_name = names_map.get(class_id, str(class_id))
                        counter[class_name] += 1

        return dict(counter)

    def group_classes_by_label_count(self) -> dict[int, list[str]]:
        # Agrupa clases que tienen la misma cantidad de labels.

        class_counts = self._label_counts_from_files()
        grouped: dict[int, list[str]] = defaultdict(list)

        for class_name, count in class_counts.items():
            grouped[count].append(class_name)

        return dict(grouped)

    def all_sessions_used(self) -> set[str]:
        """Retorna el conjunto de todas las sesiones presentes en los splits."""

        sessions: set[str] = set()

        for split in self.splits:
            images_dir = self.dataset_path / "split" / split / "images"

            for img_path in images_dir.iterdir():
                if not img_path.is_file():
                    continue

                # Usa la misma lógica de nombres que el split (parse_filename)
                _, session = parse_filename(img_path.name)
                sessions.add(session)

        return sessions
    def summary_split(self) -> dict:
        counts = self.count_images_per_split()
        percentages = self.compute_split_percentages()

        # total de labels por clase
        class_counts = self.total_labels_per_class()

        # productos en split con misma cantidad de imágenes
        grouped = self.group_classes_by_label_count()

        print("=== Resumen de splits ===")
        for split in self.splits:
            count = counts.get(split, 0)
            pct = percentages.get(split, 0.0)
            print(f"- {split}: {count} imágenes ({pct}%)")

        print("\n=== Total de labels por clase ===")
        for class_name, count in class_counts.items():
            print(f"- {class_name}: {count} labels")

        print("\n=== Clases agrupadas por cantidad de labels ===")
        for count, class_names in grouped.items():
            joined = ", ".join(sorted(class_names))
            print(f"- {count} labels: {joined}")
