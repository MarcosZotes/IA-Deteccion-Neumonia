import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = todos, 1 = info, 2 = warning, 3 = error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# === CONFIGURACIÓN GENERAL ===
LEARNING_RATE = 1e-4
base_path = './Testeo/Comparativas'
matriz_conf_path = os.path.join(base_path, 'MatrizConfusion')
errores_path = os.path.join(base_path, 'PrediccionesIncorrectas')
os.makedirs(os.path.join(base_path, 'Metricas'), exist_ok=True)
os.makedirs(matriz_conf_path, exist_ok=True)
os.makedirs(errores_path, exist_ok=True)

# === DATOS DE VALIDACIÓN ===
X_val = np.load('./AnalisisDatos/val/X.npy').astype("float32")
y_val = np.load('./AnalisisDatos/val/y.npy')

def cargar_configuracion(id_entrenamiento):
    ruta = f'./Entrenamiento/Config_Entrenamiento/config_{id_entrenamiento}.json'
    if os.path.exists(ruta):
        with open(ruta, 'r') as f:
            config = json.load(f)
            return config.get("Tamaño_imagen", 224), config.get("Batch_size", 32)
    return 224, 32

def sparse_categorical_focal_loss(gamma=2., alpha=0.25):
    if alpha is None:
        alpha = [1.0] * 3
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true_one_hot = K.one_hot(y_true, num_classes=K.shape(y_pred)[-1])
        cross_entropy = K.categorical_crossentropy(y_true_one_hot, y_pred)
        probs = K.sum(y_true_one_hot * y_pred, axis=-1)
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        alpha_weight = tf.reduce_sum(y_true_one_hot * alpha_tensor, axis=-1)
        focal_weight = K.pow(1. - probs, gamma)
        return alpha_weight * focal_weight * cross_entropy
    return loss

def obtener_ultimos_modelos(carpeta='./ModelosGuardados', cantidad=50, filtro_ids=None):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.keras') and f.startswith('prueba_modelo')]
    if filtro_ids:
        archivos = [f for f in archivos if any(id_ in f for id_ in filtro_ids)]
    archivos_con_fecha = [(f, os.path.getmtime(os.path.join(carpeta, f))) for f in archivos]
    archivos_ordenados = sorted(archivos_con_fecha, key=lambda x: x[1], reverse=True)
    return [os.path.join(carpeta, nombre) for nombre, _ in archivos_ordenados[:cantidad]]

# === COMPARADOR DE MODELOS ===
resultados = []
ultimos_modelos = obtener_ultimos_modelos()

for modelo_path in ultimos_modelos:
    nombre_archivo = os.path.basename(modelo_path)
    id_entrenamiento = 'Entrenamiento_' + nombre_archivo.split('_')[-1].replace('.keras', '')
    # Extraer arquitectura desde el JSON de configuración
    config_path = f"./Entrenamiento/Config_Entrenamiento/config_{id_entrenamiento}.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = json.load(f)
            arquitectura = conf.get("Modelo", "Desconocida")
    else:
        arquitectura = "Desconocida"
        
    nombre_modelo = f"modelo_{id_entrenamiento}"

    try:
        config_img_size, config_batch_size = cargar_configuracion(id_entrenamiento)
        model_tmp = keras.models.load_model(modelo_path, compile=False)
        input_shape = model_tmp.input_shape[1:3]
        IMG_SIZE = input_shape[0]
        BATCH_SIZE = config_batch_size if config_batch_size < len(X_val) else 32
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
            .map(lambda img, label: (tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), label)) \
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model_tmp.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
            loss=sparse_categorical_focal_loss(alpha=0.25),
            metrics=['accuracy']
        )

        start_time = time.time()
        y_pred_prob = model_tmp.predict(val_dataset)
        end_time = time.time()
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_val_bin = keras.utils.to_categorical(y_val, num_classes=3)

        auc = roc_auc_score(y_val_bin, y_pred_prob, multi_class="ovr")
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        macro_recall = recall_score(y_val, y_pred, average='macro')
        macro_precision = precision_score(y_val, y_pred, average='macro')

        resultado = {
            "Nombre_Modelo": nombre_modelo,
            "Arquitectura": arquitectura,
            "Val_Accuracy": np.mean(y_pred == y_val),
            "AUC": auc,
            "Tiempo_Prediccion_s": round(end_time - start_time, 2),
            "Macro_F1": macro_f1,
            "Macro_Recall": macro_recall,
            "Macro_Precision": macro_precision
        }
        for clase in ['0', '1', '2']:
            resultado[f'Precision_Clase_{clase}'] = report[clase]['precision']
            resultado[f'Recall_Clase_{clase}'] = report[clase]['recall']
            resultado[f'F1_Clase_{clase}'] = report[clase]['f1-score']

        resultados.append(resultado)
        
        with open(os.path.join(base_path, 'Metricas', f'{nombre_modelo}_resumen.json'), 'w') as json_f:
            json.dump(resultado, json_f, indent=4)

    except Exception as e:
        print(f"[ERROR] {nombre_modelo}: {e}")
        continue

# === EXPORTAR RESULTADOS ===
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(os.path.join(base_path, 'Metricas', 'comparativa_ultimos_modelos.csv'), index=False)

print("\nTOP 3 modelos por Macro F1:")
top3 = df_resultados.sort_values(by='Macro_F1', ascending=False).head(5)
for i, row in top3.iterrows():
    print(f"#{i+1} → {row['Nombre_Modelo']} ({row['Arquitectura']})")
    print(f"     F1: {row['Macro_F1']:.4f} | Acc: {row['Val_Accuracy']:.4f} | AUC: {row['AUC']:.4f} | Tiempo: {row['Tiempo_Prediccion_s']}s")


