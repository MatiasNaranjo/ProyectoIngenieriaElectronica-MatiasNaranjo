from src.utils.files import (
    exportar_a_raspberry,
    importar_de_raspberry,
    list_files_by_prefix,
)


def deploy_raspi(config):
    # Copiar archivos de la PC a la carpeta principal de Raspberry PI
    exportar_a_raspberry(
        local_dir=config.pc.dir_raspi,
        remote_dir=config.raspi.dir_base,
        raspberry_user=config.raspi.user,
        raspberry_ip=config.raspi.ip,
        files=config.pc.files_up,
        folders=config.pc.folders_up,
        key_path=config.pc.key_path,
        passphrase=config.pc.passphrase,
    )

    # Copiar archivos de configuración PC a Raspberry PI
    list_config = list_files_by_prefix(config.pc.dir_config, config.pc.prefix_config)
    exportar_a_raspberry(
        local_dir=config.pc.dir_config,
        remote_dir=config.raspi.dir_config,
        raspberry_user=config.raspi.user,
        raspberry_ip=config.raspi.ip,
        files=list_config,
        key_path=config.pc.key_path,
        passphrase=config.pc.passphrase,
    )

    # Copiar archivos de types PC a Raspberry PI
    list_types = list_files_by_prefix(config.pc.dir_types, config.pc.prefix_types)
    exportar_a_raspberry(
        local_dir=config.pc.dir_types,
        remote_dir=config.raspi.dir_types,
        raspberry_user=config.raspi.user,
        raspberry_ip=config.raspi.ip,
        files=list_types,
        key_path=config.pc.key_path,
        passphrase=config.pc.passphrase,
    )

    # Copiar config_loader PC a Raspberry PI
    exportar_a_raspberry(
        local_dir=config.pc.dir_utils,
        remote_dir=config.raspi.dir_utils,
        raspberry_user=config.raspi.user,
        raspberry_ip=config.raspi.ip,
        files=config.pc.file_config_loader,
        key_path=config.pc.key_path,
        passphrase=config.pc.passphrase,
    )

    # Copiar archivos de RaspberryPI a la PC
    importar_de_raspberry(
        remote_dir=config.raspi.dir_base,
        local_dir=config.pc.dir_base,
        raspberry_user=config.raspi.user,
        raspberry_ip=config.raspi.ip,
        folders=config.raspi.folders_down,
        key_path=config.pc.key_path,
        passphrase=config.pc.passphrase,
    )
