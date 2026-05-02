from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ── Tipos de error ──────────────────────────────────────────────────────────
ERROR_FN = "false_negative"  # GT sin predicción (el modelo no detectó nada)
ERROR_FP = "false_positive"  # Predicción sin GT (el modelo inventó una detección)
ERROR_WRONG_CLASS = "wrong_class"  # Bounding box correcto pero clase equivocada


def _load_class_names(dataset_path: Path) -> dict[int, str]:
    """Carga el mapping id→nombre desde dataset_meta.yaml o data.yaml."""
    for candidate in ["dataset_meta.yaml", "data.yaml"]:
        meta = dataset_path / candidate
        if not meta.exists():
            continue
        data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
    return {}


def _parse_label_file(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    Lee un .txt en formato YOLO y devuelve lista de boxes en xyxy absoluto.

    Formato YOLO: class_id  x_center  y_center  width  height  (normalizados 0-1)
    """
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) == 4:
            # Formato detection: x_center y_center width height
            xc, yc, w, h = coords
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
        else:
            # Formato segmentación: x1 y1 x2 y2 ... xn yn (normalizados)
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        boxes.append({"class_id": cls_id, "bbox": (x1, y1, x2, y2)})

    return boxes


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Calcula IoU entre dos cajas en formato xyxy."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_threshold: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Empareja predicciones con ground-truth usando IoU greedy.

    Retorna:
        wrong_class : pares (gt, pred) donde IoU >= umbral pero clase distinta
        false_neg   : boxes GT sin predicción asociada
        false_pos   : predicciones sin GT asociada
    """
    matched_gt = set()
    matched_pred = set()
    wrong_class = []

    # Construir matriz de IoU
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = _compute_iou(gt["bbox"], pred["bbox"])

    # Asignación greedy de mayor a menor IoU
    for _ in range(min(len(gt_boxes), len(pred_boxes))):
        if iou_matrix.size == 0:
            break
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        if iou_matrix[i, j] < iou_threshold:
            break

        gt = gt_boxes[i]
        pred = pred_boxes[j]

        if gt["class_id"] != pred["class_id"]:
            wrong_class.append({"gt": gt, "pred": pred})

        matched_gt.add(i)
        matched_pred.add(j)

        # Invalidar fila y columna para no reusar el mismo box
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    false_neg = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    false_pos = [pred_boxes[j] for j in range(len(pred_boxes)) if j not in matched_pred]

    return wrong_class, false_neg, false_pos


# Colores BGR por tipo de box en la imagen guardada
_COLOR = {
    "gt": (0, 255, 0),  # Verde  → ground truth
    "pred_ok": (255, 165, 0),  # Naranja → predicción correcta (no debería aparecer)
    "fp": (0, 0, 255),  # Rojo   → falso positivo
    "fn": (128, 0, 128),  # Violeta→ falso negativo
    "wc_gt": (255, 255, 0),  # Amarillo → GT del wrong class
    "wc_pred": (0, 128, 255),  # Celeste  → pred del wrong class
}


class PredictionAnalyzer:
    """
    Analiza los errores de un modelo YOLO sobre un dataset spliteado.
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.class_names = _load_class_names(self.dataset_path)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, str(class_id))

    def _predict(self, img_path: Path) -> list[dict]:
        """Corre inferencia y devuelve boxes en xyxy absoluto."""
        results = self.model.predict(
            str(img_path),
            conf=self.conf_threshold,
            verbose=False,
        )
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return detections

    def _draw_and_save(
        self,
        img_path: Path,
        gt_boxes: list[dict],
        pred_boxes: list[dict],
        wrong_class: list[dict],
        false_neg: list[dict],
        false_pos: list[dict],
        output_dir: Path,
    ) -> None:
        """Dibuja GT y predicciones con código de colores y guarda la imagen."""
        frame = cv2.imread(str(img_path))
        if frame is None:
            return

        def _draw_box(box_dict, color, label_text):
            x1, y1, x2, y2 = map(int, box_dict["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Falsos negativos (GT sin match) → violeta
        for b in false_neg:
            _draw_box(b, _COLOR["fn"], f"FN: {self._class_name(b['class_id'])}")

        # Falsos positivos (pred sin match) → rojo
        for b in false_pos:
            _draw_box(
                b,
                _COLOR["fp"],
                f"FP: {self._class_name(b['class_id'])} {b['confidence']:.2f}",
            )

        # Wrong class → GT amarillo, pred celeste
        for pair in wrong_class:
            gt, pred = pair["gt"], pair["pred"]
            _draw_box(gt, _COLOR["wc_gt"], f"GT: {self._class_name(gt['class_id'])}")
            _draw_box(
                pred,
                _COLOR["wc_pred"],
                f"PRED: {self._class_name(pred['class_id'])} {pred['confidence']:.2f}",
            )

        # Leyenda en la esquina
        legend = [
            ("FN (no detecto)", _COLOR["fn"]),
            ("FP (deteccion falsa)", _COLOR["fp"]),
            ("GT wrong class", _COLOR["wc_gt"]),
            ("PRED wrong class", _COLOR["wc_pred"]),
        ]
        for idx, (text, color) in enumerate(legend):
            cv2.putText(
                frame, text, (8, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Guardar en subcarpeta por tipo de error
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / img_path.name), frame)

    def _print_summary(self, summary: dict) -> None:
        print("\n" + "-" * 50)
        print("RESUMEN DE ERRORES DEL MODELO")

        for split, counts in summary.items():
            print(f"\n[{split.upper()}]")
            print(f"  Imágenes con errores : {counts['total_images_with_errors']}")
            print(f"  Falsos negativos     : {counts[ERROR_FN]}")
            print(f"  Falsos positivos     : {counts[ERROR_FP]}")
            print(f"  Clase equivocada     : {counts[ERROR_WRONG_CLASS]}")
        print("-" * 50)

    # ── análisis por imagen ──────────────────────────────────────────────────

    def _analyze_image(self, img_path: Path, label_path: Path) -> dict | None:
        """
        Analiza una imagen. Retorna un dict con los errores encontrados,
        o None si la imagen no tiene ningún error.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]

        gt_boxes = _parse_label_file(label_path, w, h)
        pred_boxes = self._predict(img_path)

        wrong_class, false_neg, false_pos = _match_boxes(
            gt_boxes, pred_boxes, self.iou_threshold
        )

        has_error = wrong_class or false_neg or false_pos
        if not has_error:
            return None

        return {
            "image": img_path,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "wrong_class": wrong_class,
            "false_neg": false_neg,
            "false_pos": false_pos,
        }

    # ── análisis por split ───────────────────────────────────────────────────

    def analyze_split(self, split: str) -> list[dict]:
        """Analiza todas las imágenes de un split y retorna lista de errores."""
        images_dir = self.dataset_path / split / "images"
        labels_dir = self.dataset_path / split / "labels"

        if not images_dir.exists():
            print(f"[WARN] Split '{split}' no encontrado en {images_dir}")
            return []

        errors = []
        img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        print(f"\n[{split.upper()}] Analizando {len(img_paths)} imágenes...")

        for img_path in img_paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            result = self._analyze_image(img_path, label_path)
            if result:
                result["split"] = split
                errors.append(result)

        return errors

    # ── punto de entrada ─────────────────────────────────────────────────────
    def run(
        self,
        splits: list[str] = None,
        output_dir: str = "reports/errors",
        save_images: bool = True,
    ):
        splits = splits or ["val", "test"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_errors = []
        for split in splits:
            all_errors.extend(self.analyze_split(split))

        # Contar errores por tipo
        summary: dict[str, dict] = defaultdict(
            lambda: {
                "total_images_with_errors": 0,
                ERROR_FN: 0,
                ERROR_FP: 0,
                ERROR_WRONG_CLASS: 0,
            }
        )

        for err in all_errors:
            split = err["split"]
            summary[split]["total_images_with_errors"] += 1
            summary[split][ERROR_FN] += len(err["false_neg"])
            summary[split][ERROR_FP] += len(err["false_pos"])
            summary[split][ERROR_WRONG_CLASS] += len(err["wrong_class"])

            if save_images:
                self._draw_and_save(
                    img_path=err["image"],
                    gt_boxes=err["gt_boxes"],
                    pred_boxes=err["pred_boxes"],
                    wrong_class=err["wrong_class"],
                    false_neg=err["false_neg"],
                    false_pos=err["false_pos"],
                    output_dir=output_path / split,
                )

        self._print_summary(summary)

        return dict(summary)
