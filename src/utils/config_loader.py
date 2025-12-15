import importlib
import platform
from dataclasses import is_dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class ConfigLoader:
    def __init__(self, func_name: str):
        """
        Loader centralizado de configuración para el proyecto.

        Se encarga de:
        - Cargar y combinar configuraciones desde archivos YAML y variables de entorno.
        - Validar la configuración final mediante modelos Pydantic.
        - Determinar el directorio base del proyecto.
        - Detectar el dispositivo de ejecución (CPU / GPU).

        Este loader permite definir valores por defecto en YAML y sobrescribirlos
        de forma segura mediante variables de entorno, manteniendo una estructura
        tipada y validada.

        Atributos
        ----------
        func_name: str
            Nombre de la funcionalidad
        device: str
            Dispositivo de ejecución
        """
        self.func_name = func_name
        self.device = self.detect_device()

    def _load_config_class(self):
        # Construye dinámicamente el path del módulo de configuración
        module_path = (
            f"src.utils.config.types.config_types_{self.device}_{self.func_name}"
        )

        # Importa el módulo de configuración en tiempo de ejecución
        # Ejemplo: src.utils.config.types.config_types_pc_training
        module = importlib.import_module(module_path)

        class_name = "AppConfig"
        # Obtiene la clase AppConfig desde el módulo importado
        config_class = getattr(module, class_name)

        if not is_dataclass(config_class):
            raise TypeError(f"{class_name} tiene que se un dataclass")

        # Devuelve la clase de configuración (no la instancia)
        return config_class

    def detect_device(self):
        """
        Detecta si se está ejecutando en:
        - PC Windows → 'pc'
        - Raspberry Pi o Linux ARM → 'raspi'
        """
        return "raspi" if platform.system().lower() != "windows" else "pc"
