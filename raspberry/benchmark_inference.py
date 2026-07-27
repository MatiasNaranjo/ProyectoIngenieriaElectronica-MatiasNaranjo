import csv
import logging
import os

from src.benchmark.runner import run_benchmark, summary_stats, warmup
from src.camera.camera_utils import init_camera
from ultralytics import YOLO

from src.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main(config, picam2):
    """
    Benchmark de latencia de inferencia del modelo YOLO en Raspberry Pi.

    Captura UN frame del sensor real, cierra la cámara, y cronometra
    N llamadas a model.predict() sobre esa misma matriz en memoria
    (elimina variabilidad de captura entre iteraciones). Descarta los
    primeros `warmup` frames. Escribe CSV per-frame y loguea el resumen
    estadístico (n, mean, std, p50, p95, p99, min, max).

    Args:
        config: configuración de la aplicación cargada.
        picam2: instancia de la cámara inicializada.
    """
    logging.info("Iniciando benchmark de inferencia")

    os.makedirs(
        os.path.dirname(config.benchmark.output_csv) or ".", exist_ok=True
    )

    # Cargar el modelo
    model = YOLO(config.benchmark.model_path)
    logging.info("Modelo cargado, task: %s\n", model.task)

    # Capturar un único frame del sensor real
    frame = picam2.capture_array()
    logging.info("Frame capturado: shape=%s dtype=%s", frame.shape, frame.dtype)
    picam2.stop()

    predict_kwargs = dict(
        imgsz=config.benchmark.resolution,
        conf=config.benchmark.conf,
        iou=config.benchmark.iou,
        verbose=False,
    )

    # Warm-up (descartado)
    logging.info("Warm-up: %d frames (descartados)", config.benchmark.warmup)
    warmup(model, frame, config.benchmark.warmup, **predict_kwargs)

    # Detecciones sobre el frame (idénticas en todas las iteraciones,
    # se loguean una sola vez como control de que el modelo "ve" algo)
    results = model.predict(frame, **predict_kwargs)
    counts = {}
    for box in results[0].boxes:
        name = model.names[int(box.cls)]
        counts[name] = counts.get(name, 0) + 1
    logging.info(
        "Detecciones en el frame (total=%d): %s",
        sum(counts.values()),
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "ninguna",
    )

    # Medición
    logging.info("Midiendo %d frames...", config.benchmark.num_frames)
    times = run_benchmark(
        model, frame, config.benchmark.num_frames, **predict_kwargs
    )

    # CSV per-frame
    with open(config.benchmark.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "inference_ms"])
        w.writerows(enumerate(times))
    logging.info("CSV guardado en: %s", config.benchmark.output_csv)

    # Resumen
    s = summary_stats(times)
    logging.info(
        "n=%d  mean=%.2f  std=%.2f  p50=%.2f  p95=%.2f  p99=%.2f  "
        "min=%.2f  max=%.2f  (ms)",
        s["n"], s["mean_ms"], s["std_ms"],
        s["p50_ms"], s["p95_ms"], s["p99_ms"],
        s["min_ms"], s["max_ms"],
    )


if __name__ == "__main__":
    config = ConfigLoader(func_name="benchmark").load()
    picam2 = init_camera(resolution=config.benchmark.resolution)

    try:
        main(config, picam2)
    except KeyboardInterrupt:
        logging.info("Detenido por el usuario")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
