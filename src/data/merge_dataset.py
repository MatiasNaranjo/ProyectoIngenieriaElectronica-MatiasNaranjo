import shutil
from pathlib import Path


def find_roboflow_dataset_root(input_dir: Path) -> Path:
    # Obtener todas las subcarpetas dentro del directorio de entrada
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    # Verificar que exista exactamente una carpeta de dataset
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Se esperaba una única carpeta de dataset en {input_dir}, "
            f"pero se encontraron {len(subdirs)}"
        )

    # Devolver la carpeta raíz del dataset
    return subdirs[0]


def merge_roboflow_dataset(input_dir: str, output_dir: str) -> None:
    # Convertir las rutas de entrada y salida a objetos Path
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Encontrar la carpeta raíz del dataset Roboflow
    dataset_root = find_roboflow_dataset_root(input_dir)

    # Definir las carpetas de salida para imágenes y etiquetas
    images_out = output_dir / "merged" / "images"
    labels_out = output_dir / "merged" / "labels"

    # Crear las carpetas de salida si no existen
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Definir los splits típicos del dataset
    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = dataset_root / split

        # Si el split no existe, pasar al siguiente
        if not split_dir.exists():
            continue

        # Copiar todas las imágenes del split a la carpeta unificada
        for img in (split_dir / "images").iterdir():
            shutil.copy(img, images_out / img.name)

        # Copiar todas las etiquetas del split a la carpeta unificada
        for lbl in (split_dir / "labels").iterdir():
            shutil.copy(lbl, labels_out / lbl.name)

    print("Dataset Roboflow mergeado correctamente.")
