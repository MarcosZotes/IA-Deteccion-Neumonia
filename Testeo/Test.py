# =============================================================================
# Nombre del proyecto: Herramenta de apoyo para la detección de neumonía
# Autor: Marcos Zotes Calleja
# Universidad: Universidad Internacional de La Rioja (UNIR)
# Grado: Grado en Ingeniería Informática
# Trabajo Fin de Estudios (TFE)
# Curso académico: 2024/2025
# Fecha: 
# Versión: 1.0
#
# Descripción:
# Este script realiza el preprocesamiento de datos para el entrenamiento de modelos
# de clasificación de imágenes de rayos X. Sus funcionalidades incluyen:
#
# - Carga de imágenes desde carpetas organizadas por conjuntos (train, val, test).
# - Clasificación de imágenes en tres clases: NORMAL, PNEUMONIA_VIRAL y PNEUMONIA_BACTERIAL.
# - Balanceo automático de las clases para evitar sesgo.
# - División estratificada en conjuntos de entrenamiento, validación y test.
# - Guardado de los datos en archivos .npy para uso eficiente en modelos de IA.
# - Generación de gráficas de distribución de clases.
# - Exportación de un resumen en formato JSON para verificación posterior.
#
# Derechos de autor © 2025 Marcos Zotes Calleja. Todos los derechos reservados.
# Este código es parte del Trabajo de Fin de Estudios. Su uso o distribución requiere
# autorización expresa del autor.
# =============================================================================

# === IMPORTAR LIBRERÍAS ===
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report, roc_curve, auc
import tensorflow.keras.backend as K
from collections import Counter
import time
import shutil

# === CONFIGURACIÓN GENERAL ===
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
timestamp = "20250412_105257"
modelo_final_path = './ModelosGuardados/prueba_modelo_Entrenamiento_86733.keras'
resultados_path = './Resultados'
os.makedirs(resultados_path, exist_ok=True)

# === CARGA DE DATOS DE TEST ===
X_test = preprocess_input(np.load('./AnalisisDatos/test/X.npy').astype("float32"))
y_test = np.load('./AnalisisDatos/test/y.npy')
assert X_test.ndim == 4 and X_test.shape[-1] == 3

X_test_sel = X_test
y_test_sel = y_test

# === CREAR DATASET DE TEST ===
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_sel, y_test_sel)) \
    .map(lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === FOCAL LOSS ===
def sparse_categorical_focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true_one_hot = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        cross_entropy = K.categorical_crossentropy(y_true_one_hot, y_pred)
        probs = K.sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = K.pow(1. - probs, gamma)
        return alpha * focal_weight * cross_entropy
    return loss

# === CARGA Y COMPILACIÓN DEL MODELO ===
model = keras.models.load_model(modelo_final_path, compile=False)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
              loss=sparse_categorical_focal_loss(alpha=0.25),
              metrics=['accuracy'])

# === GUARDAR MODELO COMO "modelo_final.keras" ===
modelo_final_save_path = './Modelo Final/modelo_final.keras'
model.save(modelo_final_save_path)
print(f"[✔] Copia del modelo guardada en: {modelo_final_save_path}")

# === PREDICCIÓN Y MÉTRICAS ===
start = time.time()
y_pred_prob = model.predict(test_dataset)
end = time.time()

y_pred = np.argmax(y_pred_prob, axis=1)
y_test_bin = keras.utils.to_categorical(y_test_sel, num_classes=3)

nombre_archivo = f"resultados_test_modelo_final_{timestamp}"

print("Clases reales:", np.bincount(y_test_sel))
print("Clases predichas:", np.bincount(y_pred))

auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class="ovr")
report = classification_report(y_test_sel, y_pred, output_dict=True, zero_division=0)
macro_f1 = f1_score(y_test_sel, y_pred, average='macro')
macro_recall = recall_score(y_test_sel, y_pred, average='macro')
macro_precision = precision_score(y_test_sel, y_pred, average='macro')

resumen = {
    "Modelo": os.path.basename(modelo_final_path).replace('.keras', ''),
    "Accuracy": np.mean(y_pred == y_test_sel),
    "AUC": auc_score,
    "Tiempo_Prediccion_s": round(end - start, 2),
    "Macro_F1": macro_f1,
    "Macro_Recall": macro_recall,
    "Macro_Precision": macro_precision
}
for clase in ['0', '1', '2']:
    resumen[f'Precision_Clase_{clase}'] = report[clase]['precision']
    resumen[f'Recall_Clase_{clase}'] = report[clase]['recall']
    resumen[f'F1_Clase_{clase}'] = report[clase]['f1-score']

# === EXPORTAR CLASSIFICATION REPORT COMPLETO COMO .txt ===
reporte_txt_path = os.path.join(resultados_path, f"{nombre_archivo}_reporte.txt")
with open(reporte_txt_path, "w") as f:
    f.write(classification_report(y_test_sel, y_pred, digits=4))
print(f"[✔] Classification report exportado: {reporte_txt_path}")

df_resultado = pd.DataFrame([resumen])
nombre_archivo = f"resultados_test_modelo_final_{timestamp}"
df_resultado.to_csv(os.path.join(resultados_path, f"{nombre_archivo}.csv"), index=False)
with open(os.path.join(resultados_path, f"{nombre_archivo}.md"), "w") as f:
    f.write(df_resultado.to_markdown(index=False))
with open(os.path.join(resultados_path, f"{nombre_archivo}.tex"), "w") as f:
    f.write(df_resultado.to_latex(index=False, float_format="%.4f"))

# === MATRIZ DE CONFUSIÓN ===
matriz = confusion_matrix(y_test_sel, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Clase 0', 'Clase 1', 'Clase 2'],
            yticklabels=['Clase 0', 'Clase 1', 'Clase 2'])
plt.title("Matriz de Confusión - Test")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(resultados_path, f"matriz_confusion_test_{timestamp}.png"))
plt.close()

# === GRÁFICA DE DISTRIBUCIÓN DE ERRORES ===
errores_path = os.path.join(resultados_path, "Errores")
os.makedirs(errores_path, exist_ok=True)

# Construir DataFrame de errores reales vs predichos
errores = [(int(y_t), int(y_p)) for y_t, y_p in zip(y_test_sel, y_pred) if y_t != y_p]
conteo_errores = Counter(errores)

errores_data = pd.DataFrame([
    {"Real": r, "Pred": p, "Cantidad": v}
    for (r, p), v in conteo_errores.items()
])

errores_data["Clase"] = errores_data.apply(lambda row: f"Real {row['Real']} → Pred {row['Pred']}", axis=1)

plt.figure(figsize=(10, 6))
sns.barplot(data=errores_data, x="Cantidad", y="Clase", hue="Clase", palette="Blues_r", legend=False)
plt.xlabel("Cantidad de errores")
plt.ylabel("Clase real → predicha")
plt.title("Distribución de errores por clase real y predicha")
plt.tight_layout()
plt.savefig(os.path.join(errores_path, f"grafica_distribucion_errores_{timestamp}.png"))
plt.show()
plt.close()

# === FUNCIONES GRAD-CAM ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        base_model = model.get_layer("efficientnetb0")
        conv_layers = [layer.name for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No se encontró ninguna capa Conv2D en EfficientNetB0.")
        last_conv_layer_name = conv_layers[-1]
        print(f"[INFO] Usando la última capa convolucional detectada: {last_conv_layer_name}")

    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = layers.GlobalAveragePooling2D()(classifier_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    classifier_output = layers.Dense(3, activation='softmax')(x)
    classifier_model = keras.Model(classifier_input, classifier_output)

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
        img = img / 255.0
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, [IMG_SIZE, IMG_SIZE])
    return np.clip(jet_heatmap.numpy() * alpha + img, 0, 1)

# === SELECCIÓN ESTRATIFICADA PARA GRAD-CAM ===
gradcam_path = os.path.join(resultados_path, "GradCAM")

# Ruta para las imágenes dentro de GradCAM
img_gradcam_path = os.path.join(gradcam_path, "img_GradCAM")

# Elimina carpeta GradCAM y la subcarpeta si existen, y vuelve a crearlas
if os.path.exists(gradcam_path):
    shutil.rmtree(gradcam_path)
os.makedirs(img_gradcam_path, exist_ok=True)


indices_0 = np.where(y_test == 0)[0]
indices_1 = np.where(y_test == 1)[0]
indices_2 = np.where(y_test == 2)[0]

np.random.seed(42)
sample_idx = np.concatenate([
    np.random.choice(indices_0, size=90, replace=False),
    np.random.choice(indices_1, size=90, replace=False),
    np.random.choice(indices_2, size=90, replace=False),
    np.random.choice(len(y_test), size=330, replace=False)
]) # Total de imagenes escogidas 690

y_gradcam_real = []
y_gradcam_pred = []

for i, idx in enumerate(sample_idx, start=1):
    img = tf.convert_to_tensor(X_test[idx])
    label = int(y_test[idx])
    img_resized = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    pred_prob = model.predict(tf.expand_dims(img_resized, axis=0), verbose=0)
    pred_class = np.argmax(pred_prob)

    y_gradcam_real.append(label)
    y_gradcam_pred.append(pred_class)

    clase_pred = ['Normal', 'Neumonía Vírica', 'Neumonía Bacteriana'][pred_class]
    clase_real = ['Normal', 'Neumonía Vírica', 'Neumonía Bacteriana'][label]

    print(f"Grad-CAM {i}: Predicción = {clase_pred}, Real = {clase_real}")

    try:
        heatmap = make_gradcam_heatmap(tf.expand_dims(img_resized, axis=0), model)
        grad_cam_image = superponer_gradcam(img_resized, heatmap)

        plt.figure(figsize=(6, 6))
        plt.imshow(grad_cam_image)
        plt.axis('off')
        plt.title(f"Grad-CAM - Pred: {clase_pred} Real: {clase_real}")
        file_name = f"grad_cam_{i}_pred_{clase_pred}_real_{clase_real}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(img_gradcam_path, file_name))
        plt.close()
    except Exception as e:
        print(f"[ERROR] Grad-CAM falló para la imagen {i}: {e}")

# === GRÁFICA DE ACIERTOS Y ERRORES POR CLASE EN GRAD-CAM ===
from matplotlib.patches import Patch

if len(y_gradcam_real) > 0:
    clases = ['Clase 0', 'Clase 1', 'Clase 2']
    aciertos_clase = [0, 0, 0]
    fallos_clase = [0, 0, 0]

    for real, pred in zip(y_gradcam_real, y_gradcam_pred):
        if real == pred:
            aciertos_clase[real] += 1
        else:
            fallos_clase[real] += 1

    valores = aciertos_clase + fallos_clase
    etiquetas = [f"Aciertos {c}" for c in clases] + [f"Fallos {c}" for c in clases]
    colores = sns.color_palette("Blues", n_colors=6)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        valores,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colores,
        wedgeprops={'edgecolor': 'white'},
        textprops={'fontsize': 10, 'weight': 'bold'}
    )

    # Leyenda centrada abajo
    handles = [Patch(color=colores[i], label=etiquetas[i]) for i in range(len(etiquetas))]
    ax.legend(handles=handles, title="Clases", loc='lower center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=False, ncol=3)

    ax.set_title("Distribución de aciertos y fallos por clase (Grad-CAM)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    grafica_path = os.path.join(gradcam_path, f"grafica_aciertos_fallos_gradcam_clase_{timestamp}.png")
    plt.savefig(grafica_path, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[✔] Gráfica por clase Grad-CAM guardada en: {grafica_path}")
else:
    print("[⚠] No se generaron predicciones válidas para Grad-CAM. No se creó la gráfica.")


pdf_path = os.path.join(resultados_path, f"resumen_graficas_test_{timestamp}.pdf")
with PdfPages(pdf_path) as pdf:
    # Matriz de confusión
    matriz_fig = plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clase 0', 'Clase 1', 'Clase 2'],
                yticklabels=['Clase 0', 'Clase 1', 'Clase 2'])
    plt.title("Matriz de Confusión - Test")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    pdf.savefig(matriz_fig)
    plt.close()

    # Gráfico de errores
    errores_fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=errores_data, x="Cantidad", y="Clase", hue="Clase", palette="Blues_r", legend=False)
    plt.xlabel("Cantidad de errores")
    plt.ylabel("Clase real → predicha")
    plt.title("Distribución de errores por clase real y predicha")
    plt.tight_layout()
    pdf.savefig(errores_fig)
    plt.close()

    # Gráfico circular de aciertos/fallos por clase
    if len(y_gradcam_real) > 0:
        pie_fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            valores,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            colors=colores,
            wedgeprops={'edgecolor': 'white'},
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        handles = [Patch(color=colores[i], label=etiquetas[i]) for i in range(len(etiquetas))]
        ax.legend(handles=handles, title="Clases", loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=False, ncol=3)
        ax.set_title("Distribución de aciertos y fallos por clase (Grad-CAM)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(pie_fig)

        # Curva ROC por clase
        roc_fig = plt.figure(figsize=(8, 6))
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Clase {i} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falsos positivos')
        plt.ylabel('Verdaderos positivos')
        plt.title('Curva ROC por clase')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(roc_fig)

        plt.close()

print(f"[✔] PDF generado con las gráficas principales: {pdf_path}")


# === CURVA ROC POR CLASE ===
from sklearn.metrics import roc_curve, auc

# Colores por clase
colores = ['tab:blue', 'tab:orange', 'tab:green']
clases = ['Clase 0 (Normal)', 'Clase 1 (Neumonía Vírica)', 'Clase 2 (Neumonía Bacteriana)']

plt.figure(figsize=(8, 6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colores[i], lw=2,
             label=f'{clases[i]} (AUC = {roc_auc:.4f})')

# Diagonal de referencia
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC por clase')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

roc_path = os.path.join(resultados_path, f"curva_roc_por_clase_{timestamp}.png")
plt.savefig(roc_path)
plt.show()
plt.close()

print(f"[✔] Curva ROC por clase guardada en: {roc_path}")
