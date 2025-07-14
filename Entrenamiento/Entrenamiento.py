# =============================================================================
# Nombre del proyecto: Herramenta mediaca para la detección de tipos de neumonía basada en IA
# Nombre del archivo: Entrenamiento.py
# Autor: Marcos Zotes Calleja
# Universidad: Universidad Internacional de La Rioja (UNIR)
# Grado: Grado en Ingeniería Informática
# Trabajo Fin de Estudios (TFE)
# Curso académico: 2024/2025
# Fecha: 
# Versión: 1.0
#
# Descripción:
# Este script entrena un modelo de red neuronal convolucional con transferencia de aprendizaje
# (EfficientNetB0, B1 o B2) para clasificar imágenes de rayos X de tórax en tres clases:
# - Clase 0: Normal
# - Clase 1: Neumonía Vírica
# - Clase 2: Neumonía Bacteriana
#
# El entrenamiento incluye:
# - Fine-tuning parcial de EfficientNetB0
# - Focal Loss para manejar clases desbalanceadas
# - Aumento de datos leve
# - Registro automático (configuración)
#
# Derechos de autor © 2025 Marcos Zotes Calleja. Todos los derechos reservados.
# Este código es parte del Trabajo de Fin de Estudios. Su uso o distribución requiere
# autorización expresa del autor.
# =============================================================================

# Importación de librerías estándar y específicas para la contrucion del modelo, métricas y visualización
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ocuala avisos de bajo nivel de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import warnings
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW, RMSprop
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc


# === DEFINICIÓN DE LA FUNNCiÓN LOCAL LOSS PERSONALIZADA ===
def sparse_categorical_focal_loss(gamma=2., alpha=0.30):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true_one_hot = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        cross_entropy = K.categorical_crossentropy(y_true_one_hot, y_pred)
        probs = K.sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = K.pow(1. - probs, gamma)
        return alpha * focal_weight * cross_entropy
    return loss

# === CONFIGURACIÓN DE ENTORNO Y CARPETAS ===
os.makedirs('./Entrenamiento', exist_ok=True)
os.makedirs('./ModelosGuardados', exist_ok=True)
os.makedirs('./Entrenamiento/Config_Entrenamiento', exist_ok=True)

# === GENERACIÓN DE ID DE ENTRENAMIENTO ===
def generar_id_entrenamiento(carpeta='./Entrenamiento', carpeta_modelos='./ModelosGuardados'):
    """
    Genera un ID único asegurando que no se repita con ningún modelo ya guardado ni con el historial.
    """
    historial_path = os.path.join(carpeta, 'historial_entrenamiento.csv')
    modelos_existentes = os.listdir(carpeta_modelos)
    modelos_ids = [
        int(nombre.split('_')[-1].split('.')[0])
        for nombre in modelos_existentes
        if nombre.startswith("prueba_modelo_Entrenamiento_") and nombre.endswith(".keras")
    ]

    try:
        if os.path.exists(historial_path):
            historial = pd.read_csv(historial_path)
            ids_csv = [
                int(id_.split('_')[-1])
                for id_ in historial['ID_Entrenamiento'].dropna().unique()
                if id_.startswith('Entrenamiento_') and id_.split('_')[-1].isdigit()
            ]
        else:
            ids_csv = []

        ids_totales = set(ids_csv + modelos_ids)
        siguiente_id = max(ids_totales) + 1 if ids_totales else 1
    except Exception:
        siguiente_id = random.randint(10000, 99999)

    return f"Entrenamiento_{siguiente_id}"


ID_Entrenamiento = generar_id_entrenamiento()

# === DEFINICIÓN DE HIPERPARÁMETROS ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-5

# === CARGA Y PREPROCESAMIENTO DE DATOS ===
X_train = preprocess_input(np.load('./AnalisisDatos/train/X.npy').astype("float32"))
y_train = np.load('./AnalisisDatos/train/y.npy')
X_val = preprocess_input(np.load('./AnalisisDatos/val/X.npy').astype("float32"))
y_val = np.load('./AnalisisDatos/val/y.npy')

# Info rápida
print("Distribución y_val:", np.unique(y_val, return_counts=True))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print("Valores únicos y_train:", np.unique(y_train))

# === ARGUMENTACION LEVE EN TODAS LAS CLASES ===
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])


# === DEFINICIÓN DE FUNCIONES DE PREPROCESAMIENTO PARA DATASET ===
# Preprocesamiento (redimensionado + augmentation)
def preprocess_image_train(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = data_augmentation(image)
    return image, label

def preprocess_image_val(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label


# === CREACIÓN DE DATASETS PARA ENTRENAMIENTO Y VALIDACIÓN ===
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(preprocess_image_train).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess_image_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === TRANSFER LEARNING CON EfficientNetB0 ===
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

# Fine-tuning parcial de EfficientNetB0
for layer in base_model.layers[:75]:
    layer.trainable = False
for layer in base_model.layers[75:]:
    layer.trainable = True

# === CONSTRUCCIÓN DEL MODELO CON EfficientNetB0 Y CAPA DENSA ===
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='swish')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation='softmax')(x)

# === CREACIÓN DEL MODELO ===
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, momentum=0.9),
    loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.30),
    metrics=['accuracy']
)

# Obtener submodelo EfficientNetB0
base_model = model.get_layer("efficientnetb0")

# Buscar la última capa convolucional dentro de EfficientNetB0
conv_layers = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
if conv_layers:
    print("\n[INFO] Última capa convolucional en EfficientNetB0:", conv_layers[-1])
else:
    print("\n[ERROR] No se encontraron capas Conv2D dentro de EfficientNetB0.")

# === CALLBACKS ===
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(f'./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# === ENTRENAMIENTO DEL MODELO ===
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# === HISTORIAL ===
hist_df = pd.DataFrame(history.history)
hist_df['ID_Entrenamiento'] = ID_Entrenamiento
hist_df.to_csv('./Entrenamiento/historial_entrenamiento.csv', mode='a', index=False,
               header=not os.path.exists('./Entrenamiento/historial_entrenamiento.csv'))


configuracion = {
    "ID_Entrenamiento": ID_Entrenamiento,
    "Modelo": "EfficientNetB0 + FineTuning parcial + Focal Loss",
    "Tamaño_imagen": IMG_SIZE,
    "Batch_size": BATCH_SIZE,
    "Epochs": EPOCHS,
    "Optimizador": "RMSprop",
    "Learning_rate": LEARNING_RATE,
    "Oversampling": "No",
    "Dropout": [0.3, 0.2],
    "Augmentacion": [layer.__class__.__name__ for layer in data_augmentation.layers],
    "Loss": "sparse_categorical_focal_loss(gamma=2.0, alpha=0.30)",
    "Normalización visual": "img / 255.0",
    "Mejor_val_accuracy": max(history.history["val_accuracy"]),
    "Ruta_modelo_guardado": f"./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras",
    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Guardar en JSON
ruta_config = f'./Entrenamiento/Config_Entrenamiento/config_{ID_Entrenamiento}.json'
with open(ruta_config, 'w') as f:
    json.dump(configuracion, f, indent=4)
print(f"[✔] Configuración guardada en: {ruta_config}")


# === RESUMEN EN CONSOLA ===
print(f"\n[RESUMEN] Entrenamiento: {ID_Entrenamiento}")
print(f"  Modelo guardado en: ./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras")
