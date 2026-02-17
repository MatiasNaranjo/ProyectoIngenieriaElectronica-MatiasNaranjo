from pathlib import Path


def validate_split_ratio(split_ratio: dict[str, float]) -> None:
    # Validar que los porcentajes sumen 1.0
    total = sum(split_ratio.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Los porcentajes deben sumar 1.0")


def create_split_folders(dst_path: Path, split_ratio: dict[str, float]) -> None:
    # Crear carpetas para cada split (train, val y/o test) dentro de dst_path/split/
    for split in split_ratio.keys():
        (dst_path / "split" / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / "split" / split / "labels").mkdir(parents=True, exist_ok=True)


def split_by_session(
    input_dataset_path: Path,
    output_dataset_path: Path,
    split_ratio: dict[str, float],
) -> None:
    # Validar el split_ratio
    validate_split_ratio(split_ratio)

    # Crear carpetas para los splits
    create_split_folders(output_dataset_path, split_ratio)
