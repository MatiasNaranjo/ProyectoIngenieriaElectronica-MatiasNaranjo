from pathlib import Path


def validate_split_ratio(split_ratio: dict[str, float]) -> None:
    # Validar que los porcentajes sumen 1.0
    total = sum(split_ratio.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Los porcentajes deben sumar 1.0")


def split_by_session(
    input_dataset_path: Path,
    output_dataset_path: Path,
    split_ratio: dict[str, float],
) -> None:
    # Validar el split_ratio
    validate_split_ratio(split_ratio)
