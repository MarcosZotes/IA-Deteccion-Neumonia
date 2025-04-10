import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import json

# Hiperparámetros
IMG_SIZE = 224
DATASET_PATH = './Datos/chest_xray/'  # Contiene train, val y test

# Etiquetas
LABELS = {
    'NORMAL': 0,
    'PNEUMONIA_VIRAL': 1,
    'PNEUMONIA_BACTERIAL': 2
}

random.seed(42)
np.random.seed(42)

def cargar_y_balancear():
    data_por_clase = {etiqueta: [] for etiqueta in LABELS.values()}

    for subcarpeta in ['train', 'val', 'test']:
        for clase, etiqueta in LABELS.items():
            ruta_clase = os.path.join(DATASET_PATH, subcarpeta, 'PNEUMONIA' if 'PNEUMONIA' in clase else clase)
            if not os.path.exists(ruta_clase):
                continue

            if clase == 'PNEUMONIA_VIRAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'virus' in f.lower()]
            elif clase == 'PNEUMONIA_BACTERIAL':
                archivos = [f for f in os.listdir(ruta_clase) if 'bacteria' in f.lower()]
            else:
                archivos = os.listdir(ruta_clase)

            archivos = [f for f in archivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for nombre in archivos:
                data_por_clase[etiqueta].append(os.path.join(ruta_clase, nombre))

            print(f"[{subcarpeta.upper()}] {clase} ({etiqueta}) -> {len(archivos)} imágenes")

    min_cantidad = min(len(v) for v in data_por_clase.values())
    print(f"\nBalanceando cada clase a {min_cantidad} imágenes")

    datos = []
    for etiqueta, lista in data_por_clase.items():
        random.shuffle(lista)
        for ruta in lista[:min_cantidad]:
            datos.append((ruta, etiqueta))

    random.shuffle(datos)
    return datos

def dividir_y_guardar(datos):
    conjuntos = {'train': [], 'val': [], 'test': []}
    por_clase = {etiqueta: [] for etiqueta in LABELS.values()}

    for ruta, etiqueta in datos:
        por_clase[etiqueta].append(ruta)

    for etiqueta, lista_rutas in por_clase.items():
        total = len(lista_rutas)
        train_end = int(0.7 * total)
        val_end = train_end + int(0.2 * total)

        conjuntos['train'].extend([(ruta, etiqueta) for ruta in lista_rutas[:train_end]])
        conjuntos['val'].extend([(ruta, etiqueta) for ruta in lista_rutas[train_end:val_end]])
        conjuntos['test'].extend([(ruta, etiqueta) for ruta in lista_rutas[val_end:]])

    distribucion_json = {}

    for nombre, lista in conjuntos.items():
        X, y = [], []
        for ruta, etiqueta in lista:
            try:
                img = cv2.imread(ruta)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(etiqueta)
            except Exception as e:
                print(f"❌ Error con {ruta}: {e}")

        X = np.array(X)
        y = np.array(y)

        os.makedirs(f'./AnalisisDatos/{nombre.upper()}', exist_ok=True)
        np.save(f'./AnalisisDatos/{nombre.upper()}/X.npy', X)
        np.save(f'./AnalisisDatos/{nombre.upper()}/y.npy', y)
        print(f"✅ {nombre.upper()}: {len(X)} imágenes")

        # Guardar distribución para el JSON
        clases, conteo = np.unique(y, return_counts=True)
        distribucion_json[nombre.upper()] = {
            'total': int(len(y)),
            'clases': {str(clase): int(num) for clase, num in zip(clases, conteo)}
        }

        # Gráfica
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

    # Guardar JSON de distribución
    ruta_json = './AnalisisDatos/division_datos.json'
    with open(ruta_json, 'w') as f:
        json.dump(distribucion_json, f, indent=4)

    print(f"\nArchivo de distribución guardado en: {ruta_json}")

if __name__ == "__main__":
    datos = cargar_y_balancear()
    dividir_y_guardar(datos)
