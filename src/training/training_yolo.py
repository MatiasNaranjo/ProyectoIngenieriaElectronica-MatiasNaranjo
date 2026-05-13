from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml
from ultralytics import YOLO


def entrenar_yolo(
    yaml_path=None,
    imgsz=1440,
    batch=2,
    epochs=300,
    device=0,
    amp=False,
    workers=0,
    cache=False,
    close_mosaic=10,
    cos_lr=False,
    name=None,
    project=None,
):
    """
    Entrena un modelo YOLO en un dataset descargado desde Roboflow.

    Parámetros:
        base_path (str | Path): ruta base donde se encuentra data/yolo
        imgsz (int): tamaño de la imagen
        batch (int): tamaño de batch
        epochs (int): número de epochs
        device (int | str): GPU a usar ('0' o 'cpu')
        amp (bool): si usar mixed precision
        workers (int): número de workers para dataloader
        cache (str | bool): si usar cache para dataloader ('ram' o 'disk')
        close_mosaic (int): número de epochs para cerrar mosaic augmentation
        cos_lr (bool): si usar learning rate scheduler con decaimiento cosenoidal


    Returns:
        model: objeto YOLO entrenado
    """
    yaml_path = Path(yaml_path)

    # Verifico si PyTorch, CUDA y torchvision están instalados correctamente
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    print(torchvision.__version__)

    # Cargo el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")

    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML en {yaml_path}")

    # Configuro y comienzo el entrenamiento del modelo YOLO
    model.train(
        data=yaml_path,
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        workers=workers,
        device=device,
        verbose=True,
        amp=amp,
        cache=cache,
        close_mosaic=close_mosaic,
        cos_lr=cos_lr,
        name=name,
        project=project,
    )

    return model


def _extract_metrics(results_csv: Path) -> dict:
    """
    Extrae métricas del best y last epoch desde el results.csv de un run de YOLO.

    Columnas relevantes en results.csv:
        metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    """
    if not results_csv.exists():
        return {}

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    metrics_cols = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)",
    }

    # Verificar que las columnas existen
    for col in metrics_cols.values():
        if col not in df.columns:
            print(f"[WARN] Columna no encontrada en results.csv: {col}")
            return {}

    # Best epoch: el que tiene mayor mAP50-95
    best_idx = df[metrics_cols["mAP50_95"]].idxmax()
    best_row = df.loc[best_idx]

    # Last epoch: última fila
    last_row = df.iloc[-1]

    def _extract_row(row) -> dict:
        return {
            "precision": round(float(row[metrics_cols["precision"]]), 4),
            "recall": round(float(row[metrics_cols["recall"]]), 4),
            "mAP50": round(float(row[metrics_cols["mAP50"]]), 4),
            "mAP50_95": round(float(row[metrics_cols["mAP50_95"]]), 4),
        }

    return {
        "best_epoch": int(best_idx) + 1,
        "best": _extract_row(best_row),
        "last": _extract_row(last_row),
    }


def _count_train_images(data_yaml: Path) -> int:
    """Cuenta las imágenes de train en el data.yaml de un experimento."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train_path = Path(data["path"]) / data["train"]
    if not train_path.exists():
        return -1
    return len(list(train_path.glob("*.jpg")) + list(train_path.glob("*.png")))


def _get_unique_output_dir(output_dir: Path) -> Path:
    """Si output_dir ya existe, agrega sufijo numérico: experiments, experiments2, experiments3..."""
    if not output_dir.exists():
        return output_dir

    counter = 2
    while True:
        candidate = output_dir.parent / f"{output_dir.name}{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _get_exp_number(d):
    # Extrae el número del experimento del nombre del directorio (ej: exp_25 → 25)
    return int(d.name.split("_")[1])


def entrenar_experimentos(
    experiments_dir: str,
    output_dir: str,
    model="yolov8n.pt",
    imgsz=960,
    batch=2,
    epochs=200,
    device=0,
    amp=False,
    workers=0,
    cache=False,
    close_mosaic=10,
    cos_lr=False,
    train_ratios: list[float] | None = None,  # None → corre todos
) -> None:
    """
    Entrena un modelo YOLO por cada experimento en experiments_dir en serie,
    y genera un experiment_summary.yaml con métricas de best y last epoch.

    Estructura esperada de experiments_dir:
        experiments_dir/
            exp_25/data.yaml
            exp_50/data.yaml
            exp_100/data.yaml

    Estructura de salida en output_dir:
        output_dir/
            exp_25/   ← run de YOLO
            exp_50/
            exp_100/
            experiment_summary.yaml

    Parámetros:
        experiments_dir (str): carpeta con los experimentos generados por split_experiment()
        output_dir (str): carpeta raíz donde se guardan los runs y el summary
    """

    experiments_dir = Path(experiments_dir)
    output_dir = _get_unique_output_dir(Path(output_dir))
    print(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_ratios is not None:
        # Filtrar experimentos por las proporciones especificadas
        exp_dirs = [
            d
            for d in experiments_dir.iterdir()
            if d.is_dir()
            and int(d.name.split("_")[1]) in [int(r * 100) for r in train_ratios]
        ]

    else:
        # Detectar experimentos disponibles ordenados por nombre
        exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]

    # Ordenamos independientemente de si hubo filtro o no
    exp_dirs = sorted(exp_dirs, key=_get_exp_number)

    if not exp_dirs:
        raise FileNotFoundError(f"No se encontraron experimentos en {experiments_dir}")

    print(f"\nExperimentos encontrados: {[d.name for d in exp_dirs]}")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name  # ej: exp_25
        data_yaml = exp_dir / "data.yaml"

        if not data_yaml.exists():
            print(f"[WARN] No se encontró data.yaml en {exp_dir}, saltando...")
            continue

        print(f"\n{'=' * 50}")
        print(f"Entrenando: {exp_name}")
        print(f"{'=' * 50}")

        entrenar_yolo(
            yaml_path=data_yaml,
            imgsz=imgsz,
            batch=batch,
            epochs=epochs,
            device=device,
            amp=amp,
            workers=workers,
            cache=cache,
            close_mosaic=close_mosaic,
            cos_lr=cos_lr,
            name=exp_name,
            project=str(output_dir),
        )


def summarize_experiments(output_dir: str) -> None:
    """
    Lee los runs existentes en output_dir y genera experiment_summary.yaml.
    Útil para regenerar el summary sin reentrenar.
    """
    output_dir = Path(output_dir)

    exp_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")],
        key=_get_exp_number,
    )

    if not exp_dirs:
        raise FileNotFoundError(f"No se encontraron runs en {output_dir}")

    summary = {}

    for exp_dir in exp_dirs:
        results_csv = exp_dir / "results.csv"
        metrics = _extract_metrics(results_csv)
        if not metrics:
            print(f"[WARN] No se encontró results.csv en {exp_dir}, saltando...")
            continue

        ratio = int(exp_dir.name.split("_")[1]) / 100
        summary[exp_dir.name] = {
            "ratio": ratio,
            **metrics,
        }

    summary_path = output_dir / "experiment_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    print(f"Summary guardado en: {summary_path}")
