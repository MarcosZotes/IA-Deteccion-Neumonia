import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.efficientnet import preprocess_input

# === FUNCIÓN DE FOCAL LOSS PERSONALIZADA ===
def sparse_categorical_focal_loss(gamma=2., alpha=0.25):
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

# === FUNCIÓN PARA CARGAR EL MODELO ===
def cargar_modelo(ruta_modelo):
    return tf.keras.models.load_model(
        ruta_modelo,
        custom_objects={'loss': sparse_categorical_focal_loss(gamma=2.0, alpha=0.25)},
        compile=False
    )

# === FUNCIÓN PARA PREPROCESAR IMAGEN ===
def mejorar_imagen(ruta_imagen, target_size=(224, 224)):
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"[❌] La ruta no existe: {ruta_imagen}")
    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize(target_size)
    img_np = np.array(img)

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_np = preprocess_input(img_np.astype("float32"))
    return np.expand_dims(img_np, axis=0)

# === FUNCIÓN PARA PREDECIR ===
def predecir_imagen(modelo, ruta_imagen):
    img_array = mejorar_imagen(ruta_imagen)
    predicciones = modelo.predict(img_array, verbose=0)
    pred_vector = predicciones[0]
    clases = ['Normal', 'Neumonía Vírica', 'Neumonía Bacteriana']
    indice = np.argmax(pred_vector)
    confianza = float(pred_vector[indice]) * 100
    return clases[indice], round(confianza, 2)
