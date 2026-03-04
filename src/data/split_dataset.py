import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import yaml

from src.utils.files import parse_filename


def validate_split_ratio(split_ratio: dict[str, float]) -> None:
    # Validar que los porcentajes sumen 1.0
    total = sum(split_ratio.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Los porcentajes deben sumar 1.0")


def create_split_folders(dst_path: Path, session_to_split: dict[str, str]) -> None:
    # Crear carpetas para cada split (train, val y/o test) dentro de dst_path/split/

    split_root = dst_path / "split"
    # Si ya existe, eliminarlo completamente
    if split_root.exists():
        shutil.rmtree(split_root)

    # Obtener los splits únicos a crear
    splits = set(session_to_split.values())

    for split in splits:
        (split_root / split / "images").mkdir(parents=True, exist_ok=True)
        (split_root / split / "labels").mkdir(parents=True, exist_ok=True)


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


def copy_split_files(
    source_root: Path,
    destination_root: Path,
    session_to_split: Dict[str, str],
) -> None:
    source_images = source_root / "images"
    source_labels = source_root / "labels"

    # Crear estructura destino train/val/test
    create_split_folders(destination_root, session_to_split)

    for img_path in source_images.iterdir():
        # Si no es un archivo, lo ignora
        if not img_path.is_file():
            continue

        # Tomar la sesión del nombre del archivo
        _, session = parse_filename(img_path.name)

        # Buscar a qué split fue asignada esa sesión
        split = session_to_split.get(session)
        if split is None:
            continue

        # Si la sesión no tiene label, se ignora
        label_source = source_labels / f"{img_path.stem}.txt"
        if label_source.exists():
            # Copiar imagen y label a su carpeta correspondiente
            dest_img = destination_root / "split" / split / "images" / img_path.name
            dest_label = (
                destination_root / "split" / split / "labels" / label_source.name
            )
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_source, dest_label)


def generate_train_yaml(
    yaml_original: str,
    dataset_root: str,
    output_yaml: str | None = None,
):
    """
    Genera un data.yaml listo para entrenamiento YOLO
    agregando path, train, val y test según existan.

    Si output_yaml es None, sobreescribe yaml_original.
    """

    yaml_original = Path(yaml_original)
    dataset_root = Path(dataset_root)

    if output_yaml is None:
        output_yaml = yaml_original
    else:
        output_yaml = Path(output_yaml)

    # Leer el YAML original para obtener nc y names
    with open(yaml_original, "r") as f:
        data = yaml.safe_load(f)

    # Mantengo nc y names
    new_yaml = {
        "names": data["names"],
        "nc": data["nc"],
        "path": str(dataset_root.resolve() / "split"),
        "train": "train/images",
        "val": "val/images",
    }

    # Si existe test lo agrego
    if (dataset_root / "test/images").exists():
        new_yaml["test"] = "test/images"

    # Guardar el nuevo YAML en output_yaml
    with open(output_yaml, "w") as f:
        yaml.safe_dump(new_yaml, f, sort_keys=False)

    print(f"YAML de entrenamiento generado en: {output_yaml}")


def split_by_session(
    input_dataset_path: str,
    output_dataset_path: str,
    yaml_path: str,
    split_ratio: dict[str, float],
    seed: int = 42,
) -> None:
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)

    # Validar el split_ratio
    validate_split_ratio(split_ratio)

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

    # Copiar archivos a sus carpetas correspondientes según el split asignado a su sesión
    copy_split_files(input_path, output_path, split_sessions)

    # Generar el YAML de entrenamiento para YOLO
    generate_train_yaml(
        yaml_original=yaml_path,
        dataset_root=output_dataset_path,
    )
