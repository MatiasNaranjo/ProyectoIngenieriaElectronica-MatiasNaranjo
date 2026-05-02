import argparse

from src.camera.camera_utils import capture_photos, init_camera

from src.utils.config_loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description="Captura de imágenes para dataset")
    parser.add_argument(
        "--mode", choices=["iso", "duo", "multi"], required=True, help="Modo de captura"
    )

    parser.add_argument("--product1", type=str, help="Primer producto")
    parser.add_argument("--product2", type=str, help="Segundo producto (para duo)")
    parser.add_argument(
        "--session",
        type=int,
        help="Número de sesión manual (si no se pasa, se calcula automáticamente)",
    )
    args = parser.parse_args()

    # Cargar la configuración
    config = ConfigLoader(func_name="capture").load()

    # Inicializar la cámara
    picam2 = init_camera(resolution=config.cam.resolution)

    # -------- definir producto según modo --------
    if args.mode == "iso":
        if not args.product1:
            raise ValueError("En modo iso debes pasar --product1")

        filename_prefix = args.product1
        output_dir = f"{config.cam.output_dir}/{filename_prefix}"
        n_photos = config.cam.n_photos

    elif args.mode == "duo":
        if not args.product1 or not args.product2:
            raise ValueError("En modo duo debes pasar --product1 y --product2")

        productos = sorted([args.product1, args.product2])
        filename_prefix = "_".join(productos)

        output_dir = f"{config.cam.output_dir}/duo/{filename_prefix}"
        n_photos = config.cam.n_photos

    elif args.mode == "multi":
        filename_prefix = "multi"
        output_dir = f"{config.cam.output_dir}/multi"
        n_photos = 1

    # -------- captura --------

    # Capturar fotos
    capture_photos(
        picam2,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        n_photos=n_photos,
        delay=config.cam.delay,
        session=args.session,
    )


if __name__ == "__main__":
    main()
