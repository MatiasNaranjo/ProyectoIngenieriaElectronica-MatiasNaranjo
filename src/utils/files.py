import os

import paramiko
from scp import SCPClient


def exportar_a_raspberry(
    local_dir,
    remote_dir,
    raspberry_user,
    raspberry_ip,
    files=None,
    folders=None,
    key_path=None,
    passphrase=None,
):
    """
    Copia files y folders desde tu PC a una Raspberry Pi por SSH.

    Args:
        local_dir (str): Ruta local base.
        remote_dir (str): Ruta destino en la Raspberry Pi.
        raspberry_user (str): Usuario SSH de la Raspberry.
        raspberry_ip (str): IP o hostname de la Raspberry.
        files (list): Lista de nombres de files.
        folders (list): Lista de nombres de folders.
    """
    files = files or []
    folders = folders or []

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Conectando a la Raspberry Pi...")
    try:
        if key_path:
            key = paramiko.Ed25519Key.from_private_key_file(
                key_path, password=passphrase
            )
            ssh.connect(hostname=raspberry_ip, username=raspberry_user, pkey=key)
        else:
            ssh.connect(hostname=raspberry_ip, username=raspberry_user)

    except paramiko.AuthenticationException:
        print("ERROR: fallo de autenticación. Revisá clave y passphrase.")
        return

    except Exception as e:
        print(f"ERROR: no se pudo conectar a la Raspberry Pi: {e}")
        return

    with SCPClient(ssh.get_transport()) as scp:

        # Copiar files
        for file in files:
            ruta = os.path.join(local_dir, file)
            if os.path.isfile(ruta):
                print(f"Copiando file: {file}")
                scp.put(ruta, remote_path=remote_dir)
            else:
                print(f"file no encontrado: {file}")

        # Copiar folders
        for folder in folders:
            ruta = os.path.join(local_dir, folder)
            if os.path.isdir(ruta):
                print(f"Copiando folder: {folder}")
                scp.put(ruta, remote_path=remote_dir, recursive=True)
            else:
                print(f"Carpeta no encontrada: {folder}")

    print("¡Transferencia completada!\n")
    ssh.close()


def importar_de_raspberry(
    remote_dir,
    local_dir,
    raspberry_user,
    raspberry_ip,
    folders=None,
    key_path=None,
    passphrase=None,
):
    """
    Descarga archivos y carpetas desde una Raspberry Pi a tu PC por SSH, evitando
    volver a descargar archivos que ya existen localmente.

    Args:
    remote_dir (str): Ruta base en la Raspberry Pi desde donde se descargan los archivos/carpetas.
    local_dir (str): Ruta base local donde se guardarán los archivos descargados.
    raspberry_user (str): Usuario SSH para acceder a la Raspberry Pi.
    raspberry_ip (str): IP o hostname de la Raspberry Pi.
    folders (list): Lista de nombres de carpetas dentro de remote_dir a descargar recursivamente.
    key_path (str): Ruta al archivo de clave privada para autenticación SSH.
    passphrase (str): Contraseña para la clave privada, si está encriptada.
    """

    import posixpath

    folders = folders or []

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Conectando a la Raspberry Pi...")
    try:
        if key_path:
            key = paramiko.Ed25519Key.from_private_key_file(
                key_path, password=passphrase
            )
            ssh.connect(hostname=raspberry_ip, username=raspberry_user, pkey=key)
        else:
            ssh.connect(hostname=raspberry_ip, username=raspberry_user)

    except paramiko.AuthenticationException:
        print("ERROR: fallo de autenticación. Revisá clave y passphrase.")
        return

    except Exception as e:
        print(f"ERROR: no se pudo conectar a la Raspberry Pi: {e}")
        return

    with SCPClient(ssh.get_transport()) as scp:
        cant_exist = 0
        cant_desc = 0
        # Descargar archivos dentro de carpetas (uno por uno)
        for folder in folders:
            local_folder = os.path.join(local_dir, folder)
            remote_folder = posixpath.join(remote_dir, folder)

            # Listar archivos dentro de la carpeta remota
            stdin, stdout, stderr = ssh.exec_command(f"find {remote_folder} -type f")
            remote_files = stdout.read().decode().splitlines()

            for remote_file in remote_files:
                # Construir ruta relativa para replicar estructura
                relative_path = os.path.relpath(remote_file, remote_folder)
                local_file_path = os.path.join(local_folder, relative_path)

                if os.path.exists(local_file_path):
                    print(f"Archivo ya existe: {relative_path} — se omite.")
                    cant_exist += 1
                    continue

                print(f"Descargando: {relative_path}")
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                scp.get(remote_file, local_file_path)
                cant_desc += 1
    print(f"Se descargaron {cant_desc} archivos.")
    print(f"Hay {cant_exist} archivos que ya existen.")
    print("¡Descarga completada!\n")
    ssh.close()
