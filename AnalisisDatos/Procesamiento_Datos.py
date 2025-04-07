import os
import numpy as np
import cv2
import random

# Hiperparámetros
IMG_SIZE = 224
DATASET_PATH = './Datos/chest_xray/'  # Ruta de la carpeta que contiene train, val y test

# Etiquetas asignadas a cada clase
LABELS = {'NORMAL': 0, 'PNEUMONIA_VIRAL': 1, 'PNEUMONIA_BACTERIAL': 2}


def cargar_imagenes(directorio, etiqueta):
    X = []
    y = []

    for img_name in os.listdir(directorio):
        img_path = os.path.join(directorio, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Ignorar archivos que no sean imágenes
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(etiqueta)
        except Exception as e:
            print(f"Error al procesar la imagen {img_name}: {e}")

    return np.array(X), np.array(y)



def procesar_datos():
    conjuntos = ['train', 'val', 'test']
    datos_finales = {}

    for conjunto in conjuntos:
        X, y = [], []

        for clase, etiqueta in LABELS.items():
            ruta_clase = os.path.join(DATASET_PATH, conjunto, 'PNEUMONIA' if 'PNEUMONIA' in clase else clase)

            if clase == 'PNEUMONIA_VIRAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'virus' in f.lower()]
            elif clase == 'PNEUMONIA_BACTERIAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'bacteria' in f.lower()]
            else:  # Para la clase NORMAL
                archivos = os.listdir(ruta_clase)

            for img_name in archivos:
                img_path = os.path.join(ruta_clase, img_name)
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue  # Ignorar archivos que no sean imágenes
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(etiqueta)
                except Exception as e:
                    print(f"Error al procesar la imagen {img_name}: {e}")

        X = np.array(X)
        y = np.array(y)

        print(f"Conjunto: {conjunto} - Imágenes procesadas: {len(X)} - Etiquetas únicas: {np.unique(y)}")

        np.save(os.path.join(f'./AnalisisDatos/{conjunto}', 'X.npy'), X)
        np.save(os.path.join(f'./AnalisisDatos/{conjunto}', 'y.npy'), y)


if __name__ == "__main__":
    procesar_datos()
