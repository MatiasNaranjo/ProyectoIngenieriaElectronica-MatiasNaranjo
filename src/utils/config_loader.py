import importlib
import platform
from copy import deepcopy
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel

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

        if not issubclass(config_class, BaseModel):
            raise TypeError(f"{class_name} debe heredar de BaseModel")

        # Devuelve la clase de configuración (no la instancia)
        return config_class

    def _load_yaml(self):
        """
        Carga la configuración desde un archivo YAML específico según
        el dispositivo y la funcionalidad.

        El nombre del archivo sigue el patrón:
        config_deploy_<device>_<func_name>.yaml

        Si el archivo no existe, retorna un diccionario vacío para permitir
        el uso de valores por defecto o variables de entorno.
        """
        # Nombre dinámico del archivo de configuración
        filename = f"config_deploy_{self.device}_{self.func_name}.yaml"

        # Ruta completa al archivo YAML dentro del proyecto
        path = BASE_DIR / "config" / filename

        # Si el archivo no existe, no se rompe el flujo
        if not path.exists():
            return {}

        # Cargar YAML de forma segura
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_env(self):
        """
        Carga variables de entorno desde un archivo .env específico según
        el dispositivo y la funcionalidad.

        El nombre del archivo sigue el patrón:
        <device>_<func_name>.env

        Si el archivo no existe, retorna un diccionario vacío.
        """
        # Nombre dinámico del archivo .env
        filename = f"{self.device}_{self.func_name}.env"

        # Ruta completa al archivo .env
        path = BASE_DIR / "env" / filename

        # Si no existe el archivo, no se interrumpe el flujo
        if not path.exists():
            return {}

        # Cargar variables de entorno desde el archivo
        return dotenv_values(path)

    def _expand_env_by_prefix(self, env: dict, config_class):
        """
        Convierte variables de entorno planas en una estructura anidada
        compatible con el modelo de configuración.

        Ejemplo:
            ROBOFLOW_API_KEY=123
        Se transforma en:
            {"roboflow": {"api_key": "123"}}

        """
        structured = {}
        # Normalizar las claves del ENV
        normalized_env = {k.lower(): v for k, v in env.items()}

        # Recorrer las secciones del dataclass principal (AppConfig)
        for section_name in config_class.__annotations__.keys():
            # Construir el prefijo
            prefix = section_name.lower() + "_"
            section_data = {}

            # Buscar variables que coincidan con el prefijo
            for key, value in normalized_env.items():
                # Si empieza con el prefijo de la sección
                if key.startswith(prefix):
                    # Remover el prefijo para obtener la clave final
                    clean_key = key[len(prefix) :]
                    section_data[clean_key] = value

            # Agregar sección solo si contiene datos
            if section_data:
                structured[section_name] = section_data

        return structured

    def _deep_merge(self, base, override):
        """
        Combina recursivamente dos diccionarios.

        - Los valores de `override` tienen prioridad sobre `base`.
        - Si una clave existe en ambos diccionarios y ambos valores son dict,
        se realiza un merge profundo.
        - El diccionario base no se modifica (se trabaja sobre una copia).

        Este método es utilizado para combinar:
        - Configuración base (YAML)
        - Overrides provenientes de variables de entorno (env)
        """
        # Copia profunda para evitar mutar el diccionario original
        result = deepcopy(base)
        for key, val in override.items():
            # Si ambos valores son diccionarios, merge recursivo
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(val, dict)
            ):
                result[key] = self._deep_merge(result[key], val)
            else:
                # Caso contrario: override directo

                result[key] = val

        return result

    def detect_device(self):
        """
        Detecta si se está ejecutando en:
        - PC Windows → 'pc'
        - Raspberry Pi o Linux ARM → 'raspi'
        """
        return "raspi" if platform.system().lower() != "windows" else "pc"
