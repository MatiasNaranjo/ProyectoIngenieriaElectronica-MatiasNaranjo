from src.deploy import deploy_raspi
from src.utils.config_loader import ConfigLoader


def main():
    config = ConfigLoader("update").load()
    # Copiar archivos de la PC a Raspberry PI y de Raspberry PI a la PC
    deploy_raspi(config)


if __name__ == "__main__":
    main()
