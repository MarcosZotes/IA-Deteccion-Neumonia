# =============================================================================
# Nombre del proyecto: Herramienta médica para la detección de tipos de neumonía basada en IA
# Nombre del archivo: DistribucionOriginal.py
# Autor: Marcos Zotes Calleja
# Universidad: Universidad Internacional de La Rioja (UNIR)
# Grado: Grado en Ingeniería Informática
# Trabajo Fin de Estudios (TFE)
# Curso académico: 2024/2025
# Descripción:
# Este script genera figuras con la distribución original de clases (sin balancear)
# en los subconjuntos TRAIN, VAL y TEST del dataset.
# =============================================================================

import os
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
DATASET_PATH = './Datos/chest_xray/'  # Ruta a la carpeta raíz del dataset
CLASES_VISIBLES = ['NORMAL', 'VIRAL', 'BACTERIAL']
CLASES_DIRECTORIO = ['NORMAL', 'PNEUMONIA_VIRAL', 'PNEUMONIA_BACTERIAL']

def contar_clases(subconjunto):
    """
    Cuenta cuántas imágenes hay por clase en el subconjunto especificado (train/val/test).
    """
    conteos = []

    for clase_dir in CLASES_DIRECTORIO:
        ruta = os.path.join(DATASET_PATH, subconjunto, 'PNEUMONIA' if 'PNEUMONIA' in clase_dir else clase_dir)
        if not os.path.exists(ruta):
            conteos.append(0)
            continue

        if clase_dir == 'PNEUMONIA_VIRAL':
            archivos = [f for f in os.listdir(ruta) if 'virus' in f.lower()]
        elif clase_dir == 'PNEUMONIA_BACTERIAL':
            archivos = [f for f in os.listdir(ruta) if 'bacteria' in f.lower()]
        else:
            archivos = [f for f in os.listdir(ruta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        conteos.append(len(archivos))

    return conteos

def generar_grafica(subconjunto, conteos):
    """
    Genera y guarda la gráfica de distribución para un subconjunto dado con estilo uniforme.
    """
    os.makedirs('./AnalisisDatos/ComprobacionDatos', exist_ok=True)

    # Estilo uniforme
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASES_VISIBLES, conteos, color='skyblue')

    ax.set_title(f'Distribución de clases en el conjunto {subconjunto.upper()}', fontsize=12)
    ax.set_xlabel('Clase')
    ax.set_ylabel('Número de imágenes')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Eliminar bordes gruesos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ajuste de diseño y guardado
    plt.tight_layout()
    output_path = f'./AnalisisDatos/ComprobacionDatos/{subconjunto.lower()}_distribucion_original.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Figura guardada: {output_path}")

def mostrar_distribuciones():
    """
    Ejecuta el proceso para TRAIN, VAL y TEST.
    """
    for subconjunto in ['train', 'val', 'test']:
        print(f"\n📊 Procesando subconjunto: {subconjunto.upper()}")
        conteos = contar_clases(subconjunto)
        for clase, cantidad in zip(CLASES_VISIBLES, conteos):
            print(f"{clase}: {cantidad} imágenes")
        generar_grafica(subconjunto, conteos)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    mostrar_distribuciones()
