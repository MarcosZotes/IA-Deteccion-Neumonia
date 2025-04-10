import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random

def sparse_categorical_focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true_one_hot = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        cross_entropy = K.categorical_crossentropy(y_true_one_hot, y_pred)
        probs = K.sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = K.pow(1. - probs, gamma)
        return alpha * focal_weight * cross_entropy
    return loss


# Silenciar logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Carpetas necesarias
os.makedirs('./Entrenamiento', exist_ok=True)
os.makedirs('./Graficas/Accuracy', exist_ok=True)
os.makedirs('./Graficas/Loss', exist_ok=True)
os.makedirs('./Graficas/AUC', exist_ok=True)
os.makedirs('./Graficas/Matrices', exist_ok=True)
os.makedirs('./Graficas/GradCAM', exist_ok=True)
os.makedirs('./ModelosGuardados', exist_ok=True)
os.makedirs('./Entrenamiento/Config_Entrenamiento', exist_ok=True)

# ID incremental
def generar_id_entrenamiento(carpeta='./Entrenamiento'):
    path = os.path.join(carpeta, 'historial_entrenamiento.csv')
    if not os.path.exists(path):
        return "Entrenamiento_1"
    historial = pd.read_csv(path)
    existentes = historial['ID_Entrenamiento'].dropna().unique()
    nums = [int(id_.split('_')[-1]) for id_ in existentes if id_.startswith('Entrenamiento_') and id_.split('_')[-1].isdigit()]
    siguiente = max(nums) + 1 if nums else 1
    return f"Entrenamiento_{siguiente}"

ID_Entrenamiento = generar_id_entrenamiento()

# Parámetros
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-4

# Cargar datos y aplicar preprocesado de EfficientNet
X_train = preprocess_input(np.load('./AnalisisDatos/train/X.npy').astype("float32"))
y_train = np.load('./AnalisisDatos/train/y.npy')
X_val = preprocess_input(np.load('./AnalisisDatos/val/X.npy').astype("float32"))
y_val = np.load('./AnalisisDatos/val/y.npy')

# Info rápida
print("Distribución y_val:", np.unique(y_val, return_counts=True))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print("Valores únicos y_train:", np.unique(y_train))

# Augmentación leve
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# Preprocesamiento (redimensionado + augmentation)
def preprocess_image_train(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = data_augmentation(image)
    return image, label

def preprocess_image_val(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

# Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(preprocess_image_train).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess_image_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === TRANSFER LEARNING CON EfficientNetB0 ===
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True


inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='swish')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4, momentum=0.9),
    loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25),
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

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(f'./ModelosGuardados/prueba_modelo_{ID_Entrenamiento}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# === ENTRENAMIENTO ===
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# === HISTORIAL ===
hist_df = pd.DataFrame(history.history)
hist_df['ID_Entrenamiento'] = ID_Entrenamiento
hist_df.to_csv('./Entrenamiento/historial_entrenamiento.csv', mode='a', index=False,
               header=not os.path.exists('./Entrenamiento/historial_entrenamiento.csv'))

# === EVALUACIÓN ===
y_pred_prob = model.predict(val_dataset)
y_pred = np.argmax(y_pred_prob, axis=1)
y_val_bin = keras.utils.to_categorical(y_val, num_classes=3)

# === CLASIFICATION REPORT DETALLADO ===
# Reporte con nombres de clases
target_names = ['Clase 0', 'Clase 1', 'Clase 2']
report = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Calcular métricas adicionales si las necesitas por separado
precision = precision_score(y_val, y_pred, average=None)
recall = recall_score(y_val, y_pred, average=None)
f1 = f1_score(y_val, y_pred, average=None)

# === GUARDAR MÉTRICAS POR CLASE EN CSV ===
metricas_clase = pd.DataFrame({
    "Clase": target_names,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
})
metricas_clase["ID_Entrenamiento"] = ID_Entrenamiento
ruta_metricas = f'./Entrenamiento/metricas_por_clase_{ID_Entrenamiento}.csv'
metricas_clase.to_csv(ruta_metricas, index=False)
print(f"[✔] Métricas por clase guardadas en: {ruta_metricas}")


# Añadir ID y AUC
auc = roc_auc_score(y_val_bin, y_pred_prob, multi_class="ovr")
report_df['ID_Entrenamiento'] = ID_Entrenamiento
report_df['AUC'] = np.nan
report_df.loc['AUC'] = [auc, np.nan, np.nan, np.nan, ID_Entrenamiento, auc]

# Guardar a CSV para análisis posterior
report_df.to_csv('./Entrenamiento/metricas_resultados_completas.csv', index=True,
                 mode='a', header=not os.path.exists('./Entrenamiento/metricas_resultados_completas.csv'))

print("[✔] Reporte detallado con precision, recall y F1-score guardado.")


# === GRÁFICAS DE ENTRENAMIENTO ===
def save_individual_graph(hist_df, metric, folder_name, prefix):
    files = os.listdir(folder_name)
    file_path = os.path.join(folder_name, f"{prefix}{len(files)}.png")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x=hist_df.index, y=metric, label=f'{metric.capitalize()} Entrenamiento')
    sns.lineplot(data=hist_df, x=hist_df.index, y=f'val_{metric}', label=f'{metric.capitalize()} Validación')
    plt.title(f'{metric.capitalize()} - Entrenamiento {ID_Entrenamiento}')
    plt.xlabel('Épocas')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(file_path)
    plt.close()
    print(f"[✔] Gráfico guardado: {file_path}")

save_individual_graph(hist_df, 'accuracy', './Graficas/Accuracy', 'comprobacion_accuracy')
save_individual_graph(hist_df, 'loss', './Graficas/Loss', 'comprobacion_loss')

# === AUC ACUMULADO ===
auc_data_path = './Graficas/AUC/auc_data.csv'
new_auc = pd.DataFrame({'ID_Entrenamiento': [ID_Entrenamiento], 'AUC': [auc]})
if os.path.exists(auc_data_path):
    auc_data = pd.read_csv(auc_data_path)
    auc_data = pd.concat([auc_data, new_auc], ignore_index=True)
else:
    auc_data = new_auc
auc_data.to_csv(auc_data_path, index=False)

plt.figure(figsize=(10, 6))
sns.lineplot(data=auc_data, x='ID_Entrenamiento', y='AUC', marker='o', errorbar=None)
plt.title('Comparación de AUC entre Entrenamientos')
plt.xlabel('ID Entrenamiento')
plt.ylabel('AUC')
plt.xticks(rotation=45, ha='right')
plt.savefig('./Graficas/AUC/comprobacion_auc.png')
plt.close()

# === PREDICCIONES VISUALES ===
for images, labels in val_dataset.take(1):
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    indices = random.sample(range(len(images)), 5)
    for i in indices:
        plt.imshow(images[i].numpy() / 255.0)
        plt.title(f"Real: {labels[i].numpy()} | Pred: {pred_classes[i]}")
        plt.axis('off')
        plt.show()

# === MATRIZ DE CONFUSIÓN (guardar + mostrar) ===
def mostrar_y_guardar_matriz_confusion(y_true, y_pred, nombre_entrenamiento):
    etiquetas = ['Clase 0', 'Clase 1', 'Clase 2']
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()

    # Crear carpeta si no existe
    ruta_guardado = "./Graficas/Matrices/"
    os.makedirs(ruta_guardado, exist_ok=True)

    # Guardar la imagen con el nombre del entrenamiento
    nombre_archivo = f"matriz_confusion_{nombre_entrenamiento}.png"
    plt.savefig(os.path.join(ruta_guardado, nombre_archivo))

    # Mostrarla en pantalla también
    plt.show() 

mostrar_y_guardar_matriz_confusion(y_val, y_pred, ID_Entrenamiento)

 # === PREDICCIONES VISUALES: ACIERTOS Y ERRORES ===
os.makedirs(f'./Graficas/Predicciones/{ID_Entrenamiento}', exist_ok=True)
for images, labels in val_dataset.take(1):
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    correct = [i for i in range(len(images)) if pred_classes[i] == labels[i].numpy()]
    wrong = [i for i in range(len(images)) if pred_classes[i] != labels[i].numpy()]
    
    for grupo, indices, nombre in zip(["acierto", "error"], [correct, wrong], ["Acierto", "Fallo"]):
        for i in random.sample(indices, min(5, len(indices))):
            plt.imshow(images[i].numpy() / 255.0)
            plt.title(f"[{nombre}] Real: {labels[i].numpy()} | Pred: {pred_classes[i]}")
            plt.axis('off')
            path = f"./Graficas/Predicciones/{ID_Entrenamiento}/{grupo}_{i}.png"
            plt.savefig(path)
            plt.close()

# Guardar un batch de validación en memoria (para Grad-CAM)
for batch in val_dataset.take(1):
    val_images_batch, val_labels_batch = batch
if val_images_batch.shape[0] < 3:
    print("[WARN] Menos de 3 imágenes disponibles para Grad-CAM. Se generarán menos mapas de calor.")


# === GRAD-CAM INTERPRETABILIDAD (por entrenamiento) ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    # Si no se pasa la capa, detectar la última Conv2D de EfficientNetB0
    if last_conv_layer_name is None:
        base_model = model.get_layer("efficientnetb0")
        conv_layers = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No se encontró ninguna capa Conv2D en EfficientNetB0.")
        last_conv_layer_name = conv_layers[-1]
        print(f"[INFO] Usando la última capa convolucional detectada: {last_conv_layer_name}")

    # Submodelo convolucional
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    # Submodelo de clasificación (las capas finales del modelo principal)
    # Desde GlobalAveragePooling2D hasta Dense(softmax)
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = layers.GlobalAveragePooling2D()(classifier_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    classifier_output = layers.Dense(3, activation='softmax')(x)
    classifier_model = keras.Model(classifier_input, classifier_output)

    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superponer_gradcam(imagen, heatmap, alpha=0.4):
    img = imagen.numpy()
    if img.max() > 1.0:
        img = img / 255.0  # Asegura que esté en rango [0, 1]

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, [IMG_SIZE, IMG_SIZE])
    superimposed_img = jet_heatmap.numpy() * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 1)  # Por si se pasa de 1.0
    return superimposed_img


# Crear carpeta específica del Grad-CAM para este entrenamiento
ruta_gradcam = f'./Graficas/GradCAM/{ID_Entrenamiento}'
os.makedirs(ruta_gradcam, exist_ok=True)

for i in range(min(3, len(val_images_batch))):
    # Prepara imagen: ya está preprocesada, solo expandimos dimensión
    img_tensor = tf.expand_dims(val_images_batch[i], axis=0)

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_tensor, model)
    cam_img = superponer_gradcam(val_images_batch[i], heatmap)

    # Guardado
    path_guardado = os.path.join(ruta_gradcam, f'gradcam_{ID_Entrenamiento}_{i}.png')
    plt.imshow(cam_img)
    plt.title(f"Grad-CAM - Real: {val_labels_batch[i].numpy()}")
    plt.axis('off')
    plt.savefig(path_guardado)
    plt.close()
    print(f"[✔] Grad-CAM guardado: {path_guardado}")


# === GUARDAR CONFIGURACIÓN DEL ENTRENAMIENTO ===
configuracion = {
    "ID_Entrenamiento": ID_Entrenamiento,
    "Modelo": "EfficientNetB0 + FineTuning parcial",
    "Tamaño_imagen": IMG_SIZE,
    "Batch_size": BATCH_SIZE,
    "Epochs": EPOCHS,
    "Learning_rate": LEARNING_RATE,
    "Optimizador": "RMSprop",
    "Loss": "Focal Loss (gamma=2.0, alpha=0.25)",
    "Augmentacion": [
        "RandomFlip (horizontal)",
        "RandomRotation (0.1)",
        "RandomZoom (0.1)",
        "RandomTranslation (0.1, 0.1)",
        "RandomContrast (0.1)"
    ],
    "Grad-CAM": True,
    "Visualización de errores y aciertos": True,
    "Normalización visual": "img / 255.0"
}

import json
ruta_config = f'./Entrenamiento/Config_Entrenamiento/config_{ID_Entrenamiento}.json'
with open(ruta_config, 'w') as f:
    json.dump(configuracion, f, indent=4)
print(f"[✔] Configuración guardada en: {ruta_config}")
