import re
from collections import defaultdict
from pathlib import Path

# Constantes del módulo
FILENAME_PATTERN = re.compile(r"^(?P<product>.*?)_(?P<session>s\d+)_")


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


def parse_filename(filename: str):
    # Verificar que el nombre del archivo sigue el patrón esperado
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"No se pudo parsear {filename}")
    return match.group("product"), match.group("session")


def get_sessions_by_product(images: list[Path]) -> dict[str, set[str]]:
    # Devuelve un diccionario:
    # producto -> conjunto de sesiones disponibles

    product_sessions = defaultdict(set)

    for img_path in images:
        product, session = parse_filename(img_path.name)
        # Agrega la sesión al conjunto del producto
        product_sessions[product].add(session)

    return product_sessions


def split_by_session(
    input_dataset_path: str,
    output_dataset_path: str,
    split_ratio: dict[str, float],
) -> None:
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)
    # Validar el split_ratio
    validate_split_ratio(split_ratio)

    # Crear carpetas para los splits
    create_split_folders(output_path, split_ratio)

    # Carpetas de imágenes y etiquetas del dataset original
    images_path = input_path / "images"
    labels_path = input_path / "labels"

    # Validar que existan las carpetas de imágenes y etiquetas
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError("El dataset debe contener carpetas 'images' y 'labels'")

    # Obtener lista de imágenes en el dataset
    images = list(images_path.glob("*.jpg"))

    # Validar que se hayan encontrado imágenes
    if not images:
        raise ValueError("No se encontraron imágenes en el dataset")

    # Obtener sesiones por producto
    sessions_by_product = get_sessions_by_product(images)
