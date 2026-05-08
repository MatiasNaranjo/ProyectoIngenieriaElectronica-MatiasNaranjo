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


def collect_sessions(
    product_sessions: dict[str, set[str]],
    mode: str = "all",
) -> set[str]:
    """
    Recolecta sesiones según el modo:
    - "common": solo sesiones presentes en todos los productos
    - "all": todas las sesiones disponibles
    """

    if not product_sessions:
        return set()

    if mode == "common":
        return set.intersection(*product_sessions.values())

    elif mode == "all":
        all_sessions = set()
        for sessions in product_sessions.values():
            all_sessions.update(sessions)
        return all_sessions

    else:
        raise ValueError("mode debe ser 'common' o 'all'")


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
        "nc": len(data["names"]),
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
    mode_sessions: str = "all",
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
    list_sessions = sorted(collect_sessions(sessions_by_product, mode=mode_sessions))

    # Asignar sesiones a splits según el split_ratio y el seed para reproducibilidad
    split_sessions = assign_sessions_to_splits(list_sessions, split_ratio, seed=seed)

    # Copiar archivos a sus carpetas correspondientes según el split asignado a su sesión
    copy_split_files(input_path, output_path, split_sessions)

    # Generar el YAML de entrenamiento para YOLO
    generate_train_yaml(
        yaml_original=yaml_path,
        dataset_root=output_dataset_path,
    )


def _select_systematic(files: list[Path], n: int) -> list[Path]:
    """
    Selecciona n archivos de forma sistemática (uniforme) sobre una lista ordenada.
    Garantiza cobertura uniforme del rango (útil para fotos de plataforma giratoria).

    Ejemplo: 24 fotos, n=6 → step=4 → índices 0, 4, 8, 12, 16, 20
    """
    total = len(files)
    if n >= total:
        return files

    step = total / n
    indices = [int(i * step) for i in range(n)]
    return [files[i] for i in indices]


def _get_iso_files_by_product_session(
    train_images_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    """
    Agrupa las imágenes ISO de train por producto y por sesión, ordenadas por nombre.

    Retorna:
        { producto: { sesion: [path1, path2, ...] } }

    Las imágenes 'multi' se excluyen.
    """
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for img_path in sorted(train_images_dir.glob("*.jpg")):
        product, session = parse_filename(img_path.name)
        if product == "multi":
            continue
        grouped[product][session].append(img_path)

    return {p: dict(sessions) for p, sessions in grouped.items()}


def _select_iso_files(
    iso_by_product_session: dict[str, dict[str, list[Path]]],
    ratio: float,
) -> list[Path]:
    """
    Para cada producto y sesión, selecciona sistemáticamente `ratio` de las fotos.
    ratio=0.25 → 25% de fotos por sesión por producto.

    Mínimo 1 foto por sesión por producto.
    """
    selected = []

    for product, sessions in iso_by_product_session.items():
        for session, files in sessions.items():
            n = max(1, round(len(files) * ratio))
            selected.extend(_select_systematic(files, n))

    return selected


def _copy_experiment_split(
    split_path: Path,
    exp_path: Path,
    train_files: list[Path],
) -> None:
    """
    Copia los archivos de un experimento a exp_path/split/.
    - train: solo los archivos en train_files (imágenes + labels correspondientes)
    - val y test: copia completa del split original
    """
    src_labels = split_path / "train" / "labels"

    # --- train ---
    (exp_path / "split" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (exp_path / "split" / "train" / "labels").mkdir(parents=True, exist_ok=True)

    for img_path in train_files:
        label_path = src_labels / f"{img_path.stem}.txt"
        shutil.copy2(img_path, exp_path / "split" / "train" / "images" / img_path.name)
        if label_path.exists():
            shutil.copy2(
                label_path, exp_path / "split" / "train" / "labels" / label_path.name
            )

    # --- val y test (copia completa) ---
    for split in ["val", "test"]:
        src_split = split_path / split
        if not src_split.exists():
            continue
        for subfolder in ["images", "labels"]:
            src_dir = src_split / subfolder
            dst_dir = exp_path / "split" / split / subfolder
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)


def split_experiment(
    split_path: str | Path,
    output_path: str | Path,
    yaml_path: str | Path,
    train_ratios: list[float],
    seed: int = 42,
) -> None:
    """
    Genera subsets de entrenamiento para análisis de sensibilidad al número de fotos.

    Para cada ratio en train_ratios:
    - Selecciona sistemáticamente ratio% de las fotos ISO de train (cobertura uniforme de ángulos)
    - Mantiene todas las fotos multi en train
    - Copia val y test intactos
    - Genera data.yaml listo para entrenar

    Estructura de salida:
        output_path/
            exp_25/split/train|val|test + data.yaml
            exp_50/split/train|val|test + data.yaml
            exp_100/split/train|val|test + data.yaml

    Args:
        split_path: ruta al split original (contiene train/val/test)
        output_path: ruta donde se crearán las carpetas de experimentos
        yaml_path: ruta al data.yaml original (para nc y names)
        train_ratios: lista de ratios, ej: [0.25, 0.50, 1.0]
        seed: no usado en systematic sampling, reservado para compatibilidad futura
    """
    split_path = Path(split_path)
    output_path = Path(output_path)
    yaml_path = Path(yaml_path)

    train_images_dir = split_path / "train" / "images"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"No se encontró train/images en {split_path}")

    # Agrupar ISO por producto y sesión
    iso_by_product_session = _get_iso_files_by_product_session(train_images_dir)

    # Obtener todas las fotos multi de train
    multi_files = sorted(train_images_dir.glob("multi_*.jpg"))

    print(
        f"\nProductos ISO encontrados en train: {list(iso_by_product_session.keys())}"
    )
    print(f"Fotos multi en train: {len(multi_files)}")

    # Limpiar experimento anterior si existe
    if output_path.exists():
        shutil.rmtree(output_path)

    for ratio in train_ratios:
        pct = int(ratio * 100)
        exp_path = output_path / f"exp_{pct}"

        # Seleccionar fotos ISO según ratio
        iso_selected = _select_iso_files(iso_by_product_session, ratio)
        train_files = iso_selected + multi_files

        print(
            f"\n[exp_{pct}] ISO seleccionadas: {len(iso_selected)} | Multi: {len(multi_files)} | Total train: {len(train_files)}"
        )

        # Copiar archivos
        _copy_experiment_split(split_path, exp_path, train_files)

        # Generar data.yaml
        generate_train_yaml(
            yaml_original=yaml_path,
            dataset_root=exp_path,
            output_yaml=exp_path / "data.yaml",
        )

    print(f"\nExperimentos generados en: {output_path}")
