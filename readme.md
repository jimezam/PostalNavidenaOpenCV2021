# Introducción

xxx

######################################################################

# Instalación del ambiente de desarrollo

## En Linux Ubuntu

Instalar Python3, el sistema de manejo de paquetes (`pip`) y el sistema de ambientes virtuales (`venv`).

```
$ sudo apt install python3 python3-pip python3-venv
```

Crear un nuevo ambiente virtual para el desarrollo de aplicaciones con OpenCV.  En Windows utilice una ruta apropiada como por ejemplo `c:\venv\opencv`.

```
$ python3 -m venv ~/.venv/opencv
```

Activar el ambiente virtual que se acabó de crear.  Nótese que una vez hecho este paso, el *prompt* del *shell* empezará a mostrar el nombre del ambiente activo.

```
$ source ~/.venv/opencv/bin/activate
```

Instalar las librerías necesarias para esta práctica: `opencv` y `numpy`.

```
$ pip3 install opencv-contrib-python numpy
```

######################################################################

# Configuración Visual Studio Code

Instalar la extensión `ms-python.python`.

Indicarle a Visual Studio Code que utilice el intérprete de Python ubicado en el ambiente virtual creado en el paso anterior: `~/.venv/opencv/bin/python3`.

######################################################################


