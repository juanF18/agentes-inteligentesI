#!/bin/bash

# Nombre del entorno virtual
VENV_DIR=".venv"

# Crear el entorno virtual
python3 -m venv $VENV_DIR

# Activar el entorno virtual
source $VENV_DIR/bin/activate

#Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

echo "Entorno virtual creado y dependencias instaladas."
echo "Para activar el entorno virtual, usa: source $VENV_DIR/bin/activate"