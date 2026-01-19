from src.utils.files import (
    exportar_a_raspberry,
    importar_de_raspberry,
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
