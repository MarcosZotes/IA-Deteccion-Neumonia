# =============================================================================
# Nombre del archivo: grad_cam.py
# Proyecto: Herramienta médica para la detección de tipos de neumonía basada en IA
# Autor: Marcos Zotes Calleja
# Universidad: Universidad Internacional de La Rioja (UNIR)
# Descripción: 
# Este script ofrece funciones modulares para la generación de mapas de calor 
# Grad-CAM (Gradient-weighted Class Activation Mapping) que permiten visualizar 
# las regiones más relevantes identificadas por una CNN en imágenes médicas.
# =============================================================================

# === IMPORTACIONES NECESARIAS ===
import matplotlib
matplotlib.use('Agg') # Evita errores de backend en entornos sin GUI
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# === FUNCIONES PRINCIPALES ===

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Genera un heatmap Grad-CAM para una imagen dada utilizando el modelo especificado.
    
    Parámetros:
        img_array (Tensor): Imagen preprocesada con shape (1, IMG_SIZE, IMG_SIZE, 3).
        model (keras.Model): Modelo completo utilizado para la predicción.
        last_conv_layer_name (str, opcional): Nombre de la última capa convolucional.
        pred_index (int, opcional): Índice específico de la clase para la que se desea Grad-CAM.

    Retorna:
        heatmap (np.array): Heatmap normalizado de Grad-CAM.
    """
    base_model = model.get_layer("efficientnetb0")

    if last_conv_layer_name is None:
        conv_layers = [layer.name for layer in base_model.layers if isinstance(layer, layers.Conv2D)]
        last_conv_layer_name = conv_layers[-1]

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = layers.GlobalAveragePooling2D()(classifier_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(3, activation='softmax')(x)
    classifier_model = tf.keras.Model(classifier_input, output)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superponer_gradcam(imagen, heatmap, alpha=0.3, IMG_SIZE=224):
    """
    Superpone un heatmap de Grad-CAM sobre la imagen original.
    
    Parámetros:
        imagen (Tensor): Imagen original con valores en rango [0, 255] o [0, 1].
        heatmap (np.array): Heatmap generado por Grad-CAM.
        alpha (float): Transparencia del heatmap.
        IMG_SIZE (int): Tamaño de imagen (alto/ancho).

    Retorna:
        cam_image (np.array): Imagen resultante con heatmap aplicado.
    """
    img = imagen.numpy()
    img = img / 255.0 if img.max() > 1 else img

    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.image.resize(jet_heatmap, [IMG_SIZE, IMG_SIZE])
    cam_image = np.clip(jet_heatmap.numpy() * alpha + img, 0, 1)
    
    return cam_image

def guardar_gradcam(imagen, heatmap, pred_class, true_class, ruta, IMG_SIZE=224):
    """
    Guarda una visualización de Grad-CAM en la ruta especificada.
    
    Parámetros:
        imagen (Tensor): Imagen original.
        heatmap (np.array): Heatmap Grad-CAM.
        pred_class (str/int): Clase predicha.
        true_class (str/int): Clase real.
        ruta (str): Ruta completa (incluyendo nombre de archivo) para guardar la imagen.
        IMG_SIZE (int): Tamaño de la imagen.
    """
    cam_image = superponer_gradcam(imagen, heatmap, IMG_SIZE=IMG_SIZE)
    plt.figure(figsize=(6, 6))
    plt.imshow(cam_image)
    plt.axis("off")
    plt.title(f"Predicción: {pred_class} | Real: {true_class}")
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Ejemplo mínimo de uso si se ejecuta directamente este script.
    # Debes adaptar las siguientes líneas según tus datos/modelo específicos:
    modelo = keras.models.load_model("./Modelo Final/prueba_modelo_Entrenamiento_86733.keras")
    img = tf.random.uniform((1, 224, 224, 3))  # reemplaza con imagen real
    heatmap = make_gradcam_heatmap(img, modelo)
    cam_image = superponer_gradcam(img[0], heatmap)

    plt.imshow(cam_image)
    plt.title("Visualización Grad-CAM de prueba")
    plt.axis('off')
    plt.show()
