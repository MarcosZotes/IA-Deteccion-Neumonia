import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
import json
import random
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers

# === CONFIGURACIÓN ===
ID_Entrenamiento = "Entrenamiento_8676"  # Sustituir por el ID que quieres analizar
modelo_path = f"./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras"
X_val = preprocess_input(np.load('./AnalisisDatos/val/X.npy').astype("float32"))
y_val = np.load('./AnalisisDatos/val/y.npy')
IMG_SIZE = 260  # Tamaño de imagen para el modelo, puede ser 224, 260 o 300
BATCH_SIZE = 16

# === DATASET VALIDACIÓN ===
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
    .map(lambda img, label: (tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), label)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === CARGA DEL MODELO ===
model = keras.models.load_model(modelo_path, compile=False)

# === PREDICCIONES Y MÉTRICAS ===
y_pred_prob = model.predict(val_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)
y_val_bin = keras.utils.to_categorical(y_val, num_classes=3)

# Reporte detallado
report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
macro_f1 = f1_score(y_val, y_pred, average='macro')
macro_precision = precision_score(y_val, y_pred, average='macro')
macro_recall = recall_score(y_val, y_pred, average='macro')
auc_value = roc_auc_score(y_val_bin, y_pred_prob, multi_class="ovr")

# === GUARDAR MÉTRICAS ===
resultados_path = "./Entrenamiento/MetricasPorEntrenamiento"
os.makedirs(resultados_path, exist_ok=True)

df_report["ID_Entrenamiento"] = ID_Entrenamiento
ruta_metricas = os.path.join(resultados_path, f"metricas_{ID_Entrenamiento}.csv")
df_report.to_csv(ruta_metricas)

print(f"[✔] Métricas guardadas en: {ruta_metricas}")


# === GUARDAR AUC ===
auc_data_path = "./Graficas/AUC/auc_data.csv"
nuevo_auc = pd.DataFrame({'ID_Entrenamiento': [ID_Entrenamiento], 'AUC': [auc_value]})
if os.path.exists(auc_data_path):
    auc_df = pd.read_csv(auc_data_path)
    auc_df = pd.concat([auc_df, nuevo_auc], ignore_index=True)
else:
    auc_df = nuevo_auc
auc_df['AUC'] = pd.to_numeric(auc_df['AUC'], errors='coerce')
auc_df.dropna().to_csv(auc_data_path, index=False)

# === GRÁFICA GLOBAL DE COMPARACIÓN DE AUC ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=auc_df, x='ID_Entrenamiento', y='AUC', marker='o', errorbar=None)
plt.title('Comparación de AUC entre Entrenamientos')
plt.xlabel('ID Entrenamiento')
plt.ylabel('AUC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Ruta para guardar la gráfica acumulada
graf_auc_path = './Graficas/AUC/comprobacion_auc.png'
plt.savefig(graf_auc_path)
plt.close()

print(f"[✔] Gráfico de comparación de AUC guardado en: {graf_auc_path}")


# === CURVAS ROC POR CLASE ===
roc_dir = f"./Graficas/AUC/{ID_Entrenamiento}"
os.makedirs(roc_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"Clase {i} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f"Curvas ROC - {ID_Entrenamiento}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

# Guarda dentro de la carpeta específica
roc_path = os.path.join(roc_dir, "roc.png")
plt.savefig(roc_path)
plt.close()

print(f"[✔] Curva ROC guardada en: {roc_path}")


# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.title(f"Matriz de Confusión - {ID_Entrenamiento}")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(f"./Graficas/Matrices/matriz_confusion_{ID_Entrenamiento}.png")
plt.close()

# === GRAD-CAM PARA 3 IMÁGENES ALEATORIAS ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    base_model = model.get_layer("efficientnetb0")
    if last_conv_layer_name is None:
        conv_layers = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        last_conv_layer_name = conv_layers[-1]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(base_model.input, last_conv_layer.output)
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

def superponer_gradcam(imagen, heatmap, alpha=0.4):
    img = imagen.numpy()
    img = img / 255.0 if img.max() > 1 else img
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, [IMG_SIZE, IMG_SIZE])
    return np.clip(jet_heatmap.numpy() * alpha + img, 0, 1)

gradcam_path = f"./Graficas/GradCAM/{ID_Entrenamiento}"
os.makedirs(gradcam_path, exist_ok=True)

for i in range(3):
    img = tf.convert_to_tensor(X_val[i])
    label = int(y_val[i])
    resized = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    pred_class = np.argmax(model.predict(tf.expand_dims(resized, 0), verbose=0))
    heatmap = make_gradcam_heatmap(tf.expand_dims(resized, 0), model)
    cam_img = superponer_gradcam(resized, heatmap)
    plt.imshow(cam_img)
    plt.axis("off")
    plt.title(f"Pred: {pred_class} | Real: {label}")
    plt.savefig(f"{gradcam_path}/gradcam_{i}.png")
    plt.close()

print(f"[✔] Análisis completo del entrenamiento {ID_Entrenamiento}")
