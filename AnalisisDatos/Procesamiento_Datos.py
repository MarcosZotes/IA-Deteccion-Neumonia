# =============================================================================
# Nombre del proyecto: Herramenta mediaca para la detección de tipos de neumonía basada en IA
# Nombre del archivo: Procesamiento_Datos.py
# Autor: Marcos Zotes Calleja
# Universidad: Universidad Internacional de La Rioja (UNIR)
# Grado: Grado en Ingeniería Informática
# Trabajo Fin de Estudios (TFE)
# Curso académico: 2024/2025
# Fecha: 
# Versión: 1.0
#
# Descripción:
# Script de procesamiento de datos. Carga imágenes de rayos X desde un dataset estructurado
# en carpetas (train/val/test), las clasifica en tres categorías (NORMAL, PNEUMONIA_VIRAL,
# PNEUMONIA_BACTERIAL), las balancea y divide equitativamente para el entrenamiento del modelo
#
# Derechos de autor © 2025 Marcos Zotes Calleja. Todos los derechos reservados.
# Este código es parte del Trabajo de Fin de Estudios. Su uso o distribución requiere
# autorización expresa del autor.
# =============================================================================

# === IMPORTAR LIBRERÍAS ===
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import json

# === CONFIGURACIÓN INICIAL ===
IMG_SIZE = 224 # Tamaño de la imagen (224x224)
DATASET_PATH = './Datos/chest_xray/'  # Contiene los datos train, val y test

# Diccionario de etiqueiasd codificadas numéricamente para cada clase
LABELS = {
    'NORMAL': 0,
    'PNEUMONIA_VIRAL': 1,
    'PNEUMONIA_BACTERIAL': 2
}

# Establecer la semilla para la reproducibilidad
# (opcional, pero recomendado para resultados consistentes)
random.seed(42)
np.random.seed(42)

# === FUNCIÓN: CARGA Y BALANCEAR DATOS ===
def cargar_y_balancear():
    """
    Carga las imágenes desde las carpetas del dataset, filtrando por tipo de neumonía.
    Luego balancea el número de imágenes por clase para evitar sesgos durante el entrenamiento.
    """
    data_por_clase = {etiqueta: [] for etiqueta in LABELS.values()}

    for subcarpeta in ['train', 'val', 'test']:
        for clase, etiqueta in LABELS.items():
            ruta_clase = os.path.join(DATASET_PATH, subcarpeta, 'PNEUMONIA' if 'PNEUMONIA' in clase else clase)
            if not os.path.exists(ruta_clase):
                continue
            
            # Filtrado especifico para las clases de neumonía
            if clase == 'PNEUMONIA_VIRAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'virus' in f.lower()]
            elif clase == 'PNEUMONIA_BACTERIAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'bacteria' in f.lower()]
            else:
                archivos = os.listdir(ruta_clase)

            # Comprobar si los archivos son imágenes válidas
            archivos = [f for f in archivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for nombre in archivos:
                data_por_clase[etiqueta].append(os.path.join(ruta_clase, nombre))

            print(f"[{subcarpeta.upper()}] {clase} ({etiqueta}) -> {len(archivos)} imágenes")

    # Balanceo de clases: se toma la clase con menos imágenes y se reduce el resto a esa cantidad
    # para evitar sesgos en el entrenamiento
    min_cantidad = min(len(v) for v in data_por_clase.values())
    print(f"\nBalanceando cada clase a {min_cantidad} imágenes")

    datos = []
    for etiqueta, lista in data_por_clase.items():
        random.shuffle(lista)
        for ruta in lista[:min_cantidad]:
            datos.append((ruta, etiqueta))

    random.shuffle(datos)
    return datos

# === FUNCIÓN: DIVIDIR Y GUARDAR LOS DATOS ===
def dividir_y_guardar(datos):
    """
    Divide los datos en subconjuntos de entrenamiento, validación y prueba (70/20/10).
    Convierte las imágenes a arrays NumPy, los guarda como archivos .npy y genera gráficos.
    También exporta la distribución en formato JSON.
    """
    conjuntos = {'train': [], 'val': [], 'test': []}
    por_clase = {etiqueta: [] for etiqueta in LABELS.values()}

    # Agrupación por etiqueta para asegurar la división equitativa
    for ruta, etiqueta in datos:
        por_clase[etiqueta].append(ruta)

    # División de los datos proporcionalmente en train, val y test
    for etiqueta, lista_rutas in por_clase.items():
        total = len(lista_rutas)
        train_end = int(0.7 * total)
        val_end = train_end + int(0.2 * total)

        conjuntos['train'].extend([(ruta, etiqueta) for ruta in lista_rutas[:train_end]])
        conjuntos['val'].extend([(ruta, etiqueta) for ruta in lista_rutas[train_end:val_end]])
        conjuntos['test'].extend([(ruta, etiqueta) for ruta in lista_rutas[val_end:]])

    distribucion_json = {}

    # Procesamiento y guardado de imágenes
    for nombre, lista in conjuntos.items():
        X, y = [], []
        for ruta, etiqueta in lista:
            try:
                img = cv2.imread(ruta) # Cargar imagen
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Redimensionar imagen
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertir a RGB
                X.append(img)
                y.append(etiqueta)
            except Exception as e:
                print(f"❌ Error con {ruta}: {e}")

        X = np.array(X)
        y = np.array(y)

        # Se guardand los arrays como archivos .npy
        # Se crea la carpeta correspondiente si no existe
        os.makedirs(f'./AnalisisDatos/{nombre.upper()}', exist_ok=True)
        np.save(f'./AnalisisDatos/{nombre.upper()}/X.npy', X)
        np.save(f'./AnalisisDatos/{nombre.upper()}/y.npy', y)
        print(f"✅ {nombre.upper()}: {len(X)} imágenes")

        # Se registra el conteno de cada clase en un JSON
        clases, conteo = np.unique(y, return_counts=True)
        distribucion_json[nombre.upper()] = {
            'total': int(len(y)),
            'clases': {str(clase): int(num) for clase, num in zip(clases, conteo)}
        }

        # === VISUALIZACIÓN DE LA DISTRIBUCIÓN POR CLASE ===
        etiquetas = ['NORMAL', 'VIRAL', 'BACTERIAL']
        plt.figure(figsize=(8, 5))
        plt.bar(etiquetas, conteo, color='skyblue')
        plt.title(f"Distribución de clases en el conjunto {nombre.upper()}")
        plt.xlabel("Clase")
        plt.ylabel("Número de imágenes")
        plt.grid(True, linestyle='--', alpha=0.5)
        os.makedirs('./AnalisisDatos/ComprobacionDatos', exist_ok=True)
        plt.savefig(f'./AnalisisDatos/ComprobacionDatos/{nombre.lower()}_distribucion.png')
        plt.close()

    # === EXPORTAR DISTRIBUCIÓN A JSON ===
    # Guardar la distribución de clases en un archivo JSON
    ruta_json = './AnalisisDatos/division_datos.json'
    with open(ruta_json, 'w') as f:
        json.dump(distribucion_json, f, indent=4)

    print(f"\nArchivo de distribución guardado en: {ruta_json}")

# === EJECUCIÓN DEL SCRIPT ===
# Si se ejecuta directamente, carga y procesa los datos
if __name__ == "__main__":
    datos = cargar_y_balancear()
    dividir_y_guardar(datos)
