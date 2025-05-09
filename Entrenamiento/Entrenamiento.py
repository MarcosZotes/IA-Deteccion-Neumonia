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
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc


# === DEFINICIÓN DE LA FUNNCiÓN LOCAL LOSS PERSONALIZADA ===
def sparse_categorical_focal_loss(gamma=2., alpha=None):
    """
    Implementación personalizada de focal loss para clasificación multiclase con etiquetas enteras.
    Penaliza más los errores en clases menos representadas, mejorando el aprendizaje en conjuntos desbalanceados.
    """
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true_one_hot = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        cross_entropy = K.categorical_crossentropy(y_true_one_hot, y_pred)
        probs = K.sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = K.pow(1. - probs, gamma)
        if isinstance(alpha, list):
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = tf.reduce_sum(alpha_tensor * tf.cast(y_true_one_hot, tf.float32), axis=-1)
        else:
            alpha_factor = alpha if alpha else 1.0
        return alpha_factor * focal_weight * cross_entropy
    return loss


# === CONFIGURACIÓN DE ENTORNO Y CARPETAS ===
os.makedirs('./Entrenamiento', exist_ok=True)
os.makedirs('./ModelosGuardados', exist_ok=True)
os.makedirs('./Entrenamiento/Config_Entrenamiento', exist_ok=True)

# === GENERACIÓN DE ID DE ENTRENAMIENTO ===
def generar_id_entrenamiento(carpeta='./Entrenamiento'):
    """
    Genera un ID incremental único para cada sesión de entrenamiento, registrando el historial en CSV.
    """
    path = os.path.join(carpeta, 'historial_entrenamiento.csv')
    if not os.path.exists(path):
        return "Entrenamiento_1"
    historial = pd.read_csv(path)
    existentes = historial['ID_Entrenamiento'].dropna().unique()
    nums = [int(id_.split('_')[-1]) for id_ in existentes if id_.startswith('Entrenamiento_') and id_.split('_')[-1].isdigit()]
    siguiente = max(nums) + 1 if nums else 1
    return f"Entrenamiento_{siguiente}"

ID_Entrenamiento = generar_id_entrenamiento()

# === DEFINICIÓN DE HIPERPARÁMETROS ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-4

# === CARGA Y PREPROCESAMIENTO DE DATOS ===
X_train = preprocess_input(np.load('./AnalisisDatos/train/X.npy').astype("float32"))
y_train = np.load('./AnalisisDatos/train/y.npy')
X_val = preprocess_input(np.load('./AnalisisDatos/val/X.npy').astype("float32"))
y_val = np.load('./AnalisisDatos/val/y.npy')

# Info rápida
print("Distribución y_val:", np.unique(y_val, return_counts=True))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print("Valores únicos y_train:", np.unique(y_train))

# Aumento de datos leve (general para todas las clases)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# === DEFINICIÓN DE FUNCIONES DE PREPROCESAMIENTO PARA DATASET ===
def preprocess_image_train(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = data_augmentation(image)
    return image, label

def preprocess_image_val(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

# === AUGMENTACIÓN Y BALANCEO DE DATOS ===
# Oversampling de clase 1 (neumonía vírica), triplicada
mask_class1 = y_train == 1
X_class1 = X_train[mask_class1]
y_class1 = y_train[mask_class1]

X_train = np.concatenate([X_train, X_class1, X_class1])
y_train = np.concatenate([y_train, y_class1, y_class1])

# === CREACIÓN DE DATASETS PARA ENTRENAMIENTO Y VALIDACIÓN ===
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(preprocess_image_train).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess_image_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === TRANSFER LEARNING CON EfficientNetB0 ===
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
for layer in base_model.layers[:75]:
    layer.trainable = False
for layer in base_model.layers[75:]:
    layer.trainable = True

# Construcción del modelo con EfficientNetB0
# y capas adicionales
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation='softmax')(x)

# Compilación con optimizador AdamW + CosineDecay y Focal Loss
lr_schedule = CosineDecay(initial_learning_rate=1e-4, decay_steps=3000)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=optimizer,
    loss=sparse_categorical_focal_loss(gamma=2.0, alpha=[0.2, 0.6, 0.2]),
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
model_checkpoint = ModelCheckpoint(f'./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# === ENTRENAMIENTO DEL MODELO ===
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[early_stopping, model_checkpoint],
                    class_weight=class_weights
                    )

# === HISTORIAL ===
hist_df = pd.DataFrame(history.history)
hist_df['ID_Entrenamiento'] = ID_Entrenamiento
hist_df.to_csv('./Entrenamiento/historial_entrenamiento.csv', mode='a', index=False,
               header=not os.path.exists('./Entrenamiento/historial_entrenamiento.csv'))


# === GENERAR Y GUARDAR CONFIGURACIÓN DEL ENTRENAMIENTO  ===
augmentacion_aplicada = {
    "Todas las clases": [layer.__class__.__name__ for layer in data_augmentation.layers]
}


configuracion = {
    "ID_Entrenamiento": ID_Entrenamiento,
    "Modelo": f"{model.layers[1].__class__.__name__} + FineTuning parcial",
    "Tamaño_imagen": IMG_SIZE,
    "Batch_size": BATCH_SIZE,
    "Epochs": EPOCHS,
    "Learning_rate": model.optimizer.get_config().get("learning_rate", str(LEARNING_RATE)),
    "Optimizador": model.optimizer.__class__.__name__,
    #"Loss": "sparse_categorical_focal_loss",
    "Loss": model.loss.__name__ if hasattr(model.loss, '__name__') else str(model.loss),
    "Augmentacion": augmentacion_aplicada,
    "MixUp": False,
    "Grad-CAM": True,
    "Visualización de errores y aciertos": True,
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
