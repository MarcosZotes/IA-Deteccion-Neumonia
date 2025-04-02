# Procesamiento de los datos para que el modelo sea capaz de entenderlos para su futuro entrenamiento, validación y testeo

import os
import numpy as np
import cv2
from tqdm import tqdm

# Directorio principal de los datos
Dataset_path = "./Datos/chest_xray/"

# Tamaño de las imaganes a procesar
img_size = 224

# Función para cargar las imágenes y etiquetas
def cargar_datos(directorio):
    x = []  # Imagenes
    y = []  # Etiquetas

    for categoria in ['NORMAL', 'PNEUMONIA']:
        etiqueta = 0 if categoria == 'NORMAL' else 1
        ruta_categoria = os.path.join(directorio, categoria)

        for archivo in tqdm(os.listdir(ruta_categoria), desc = f"Procesando {categoria}"):
            try:
                ruta_imagen = os.path.join(ruta_categoria, archivo)
                imagen = cv2.imread(ruta_imagen)  # Cargar la imagen
                imagen = cv2.resize(imagen, (img_size, img_size))  # Redimensionar la imagen
                imagen = imagen / 255.0  # Normalizar la imagen
                x.append(imagen)  # Añadir la imagen a la lista
                y.append(etiqueta)  # Añadir la etiqueta a la lista

            except Exception as e:
                print(f"Error al procesar la imagen {archivo}: {e}")
    
    X = np.array(x, dtype = 'float32')  # Convertir la lista de imágenes a un array de numpy
    y = np.array(y, dtype = 'int')  # Convertir la lista de etiquetas a un array de numpy

    return X, y  # Devolver las imágenes y etiquetas procesadas


# Procesar y guardar los datos
def procesar_y_guardar_datos(tipo):
    directorio = os.path.join(Dataset_path, tipo)  # Directorio de los datos
    X, y = cargar_datos(directorio)  # Cargar los datos

    # Crear carpeta correspondiente si no existe
    output_dir = f"./AnalisisDatos/{tipo.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    # Dividir los datos en partes más pequeñas ya que son muy grandes
    np.save(os.path.join(output_dir, "X_NORMAL.npy"), X[y == 0])
    np.save(os.path.join(output_dir, "X_PNEUMONIA.npy"), X[y == 1])
    np.save(os.path.join(output_dir, "y_NORMAL.npy"), y[y == 0])
    np.save(os.path.join(output_dir, "y_PNEUMONIA.npy"), y[y == 1])


    print(f"✅ {tipo} procesados y guardados con éxito.")


# Procesar y guardar cada conjunto de datos
procesar_y_guardar_datos('train')
procesar_y_guardar_datos('val')
procesar_y_guardar_datos('test')
    
