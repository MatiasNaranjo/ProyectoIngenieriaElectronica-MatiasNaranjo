import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

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


def assign_sessions_to_splits(
    common: Set[str], split_ratio: Dict[str, float], seed: int = 42
) -> Dict[str, Set[str]]:
    # Asigna sesiones a train/val/test según split_ratio y un seed para reproducibilidad.

    # Ordenar y mezclar las sesiones comunes de forma reproducible con el seed dado
    common_list = sorted(list(common))
    rng = random.Random(seed)
    rng.shuffle(common_list)

    # Total de sesiones comunes y extraer los ratios de cada split
    n_total = len(common_list)
    r_train = split_ratio.get("train", 0)
    r_val = split_ratio.get("val", 0)
    r_test = split_ratio.get("test", 0)

    total_ratio = r_train + r_val + r_test
    if total_ratio == 0:
        raise ValueError("Todos los split son 0")

    # Normalización de los ratios
    r_train /= total_ratio
    r_val /= total_ratio
    r_test /= total_ratio

    # Calcular el número de sesiones para train
    n_train = int(r_train * n_total)

    # El resto de sesiones se asignarán a val y test según sus proporciones relativas
    remaining = n_total - n_train

    if r_val + r_test > 0:
        # Se redondea para priorizar a val ante test
        n_val = round(remaining * r_val / (r_val + r_test))
    else:
        n_val = 0

    # Asignar cada sesión a su split correspondiente
    session_to_split = {}

    for session in common_list[:n_train]:
        session_to_split[session] = "train"

    for session in common_list[n_train : n_train + n_val]:
        session_to_split[session] = "val"

    for session in common_list[n_train + n_val :]:
        session_to_split[session] = "test"

    return session_to_split


def common_sessions(product_sessions: dict[str, set[str]]) -> set[str]:
    # Devuelve el conjunto de sesiones que están presentes en todos los productos
    if not product_sessions:
        return set()
    return set.intersection(*product_sessions.values())


def split_by_session(
    input_dataset_path: str,
    output_dataset_path: str,
    split_ratio: dict[str, float],
    seed: int = 42,
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

    # Obtener sesiones comunes a todos los productos
    common = sorted(common_sessions(sessions_by_product))

    # Asignar sesiones a splits según el split_ratio y el seed para reproducibilidad
    split_sessions = assign_sessions_to_splits(common, split_ratio, seed=seed)
