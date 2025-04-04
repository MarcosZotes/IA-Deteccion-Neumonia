# Marcos Zotes Calleja
# Universidad Internacional de la Rioja
# Trabajo de Fin de Estudios, 2024-2025

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Directorio donde se guardaron los archivos originales
DATA_PATH = './Datos/chest_xray'

# Hiperparámetros
IMG_SIZE = 224
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15



def cargar_datos_combinados():
    X = []  # Imágenes
    y = []  # Etiquetas

    for subfolder in ['train', 'val', 'test']:
        for categoria in ['NORMAL', 'PNEUMONIA']:
            etiqueta = 0 if categoria == 'NORMAL' else 1
            ruta_categoria = os.path.join(DATA_PATH, subfolder, categoria)

            for archivo in tqdm(os.listdir(ruta_categoria), desc=f'Procesando {categoria} en {subfolder}', unit='imágenes'):
                try:
                    ruta_imagen = os.path.join(ruta_categoria, archivo)
                    imagen = tf.keras.preprocessing.image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
                    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
                    imagen = imagen / 255.0  # Normalizar
                    X.append(imagen)
                    y.append(etiqueta)

                except Exception as e:
                    print(f"Error al procesar la imagen {archivo}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y


def dividir_y_guardar_datos(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - TRAIN_SPLIT), random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT)), random_state=42, stratify=y_temp)

    # Guardar los conjuntos de datos
    os.makedirs('./AnalisisDatos/TRAIN', exist_ok=True)
    os.makedirs('./AnalisisDatos/VAL', exist_ok=True)
    os.makedirs('./AnalisisDatos/TEST', exist_ok=True)

    np.save('./AnalisisDatos/TRAIN/X.npy', X_train)
    np.save('./AnalisisDatos/TRAIN/y.npy', y_train)
    np.save('./AnalisisDatos/VAL/X.npy', X_val)
    np.save('./AnalisisDatos/VAL/y.npy', y_val)
    np.save('./AnalisisDatos/TEST/X.npy', X_test)
    np.save('./AnalisisDatos/TEST/y.npy', y_test)

    # Guardar un archivo JSON con información de la división
    division_info = {
        'TRAIN': len(X_train),
        'VAL': len(X_val),
        'TEST': len(X_test)
    }

    with open('./AnalisisDatos/division_info.json', 'w') as json_file:
        json.dump(division_info, json_file, indent=4)

    print(f"\n✅ Datos divididos y guardados exitosamente.")
    print(f"TRAIN: {len(X_train)} imágenes")
    print(f"VAL: {len(X_val)} imágenes")
    print(f"TEST: {len(X_test)} imágenes")


if __name__ == "__main__":
    X, y = cargar_datos_combinados()
    dividir_y_guardar_datos(X, y)
    print("\n✅ Proceso de carga y división de datos completado.")
    
    