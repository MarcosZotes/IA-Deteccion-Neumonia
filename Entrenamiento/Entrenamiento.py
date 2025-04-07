import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# Ocultar mensajes de advertencia de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Crear carpetas necesarias si no existen
os.makedirs('./Entrenamiento', exist_ok=True)
os.makedirs('./Graficas/Accuracy', exist_ok=True)
os.makedirs('./Graficas/Loss', exist_ok=True)
os.makedirs('./Graficas/AUC', exist_ok=True)

# Generar ID único para cada entrenamiento
ID_Entrenamiento = f"Entrenamiento_{int(np.random.rand()*100000)}"

# Configuraciones
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 0.0001

# Rutas de datos
TRAIN_PATH = './AnalisisDatos/train/'
VAL_PATH = './AnalisisDatos/val/'

# Cargar datos
X_train = np.load(os.path.join(TRAIN_PATH, 'X.npy'))
y_train = np.load(os.path.join(TRAIN_PATH, 'y.npy'))
X_val = np.load(os.path.join(VAL_PATH, 'X.npy'))
y_val = np.load(os.path.join(VAL_PATH, 'y.npy'))

# Revisar la distribución de clases
unique, counts = np.unique(y_train, return_counts=True)
print("Distribución de clases en el conjunto de entrenamiento:", dict(zip(unique, counts)))

# Calcular pesos de clase
class_total = len(y_train)
class_weights = {i: class_total / (3 * count) for i, count in zip(unique, counts)}
class_weights[1] *= 1.5
min_weight = min(class_weights.values())
class_weights = {k: v / min_weight for k, v in class_weights.items()}

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.2
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Transfer Learning con EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers[:-50]:  
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Construcción del modelo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Reshape((1, 1, 1280)),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
model_checkpoint = ModelCheckpoint('./ModelosGuardados/modelo_mejorado.keras', save_best_only=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Guardar historial del entrenamiento en CSV
hist_df = pd.DataFrame(history.history)
hist_df['ID_Entrenamiento'] = ID_Entrenamiento

# Definir el archivo CSV general
csv_general_path = './Entrenamiento/historial_entrenamiento.csv'

# Verificar si el archivo ya existe y agregar encabezados si es necesario
if os.path.exists(csv_general_path):
    historial_general = pd.read_csv(csv_general_path)
    if set(hist_df.columns) != set(historial_general.columns):
        print("Advertencia: Diferencias detectadas en las columnas. Creando respaldo.")
        os.rename(csv_general_path, csv_general_path.replace('.csv', f'_backup_{ID_Entrenamiento}.csv'))

# Guardar el historial de entrenamiento en un CSV
hist_df.to_csv('./Entrenamiento/historial_entrenamiento.csv', index=False, mode='a', header=not os.path.exists('./Entrenamiento/historial_entrenamiento.csv'))

# Evaluación de métricas adicionales
y_pred = np.argmax(model.predict(X_val), axis=1)
y_val_bin = keras.utils.to_categorical(y_val, num_classes=3)
y_pred_prob = model.predict(X_val)

# Reporte de clasificación
report = classification_report(y_val, y_pred, target_names=['Clase 0', 'Clase 1', 'Clase 2'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Calcular AUC
auc = roc_auc_score(y_val_bin, y_pred_prob, multi_class="ovr")

# Definir columnas consistentes
columnas_fijas = ['precision', 'recall', 'f1-score', 'support', 'ID_Entrenamiento']
for columna in columnas_fijas:
    if columna not in report_df.columns:
        report_df[columna] = np.nan

# Añadir la métrica de AUC con columna asegurada
report_df['AUC'] = np.nan  # Añadir columna AUC si no existe
report_df.loc['AUC'] = [auc, np.nan, np.nan, np.nan, ID_Entrenamiento, auc]  # Añadir la métrica AUC

# Guardar en CSV
report_df.to_csv('./Entrenamiento/metricas_resultados_completas.csv', index=True, mode='a', header=not os.path.exists('./Entrenamiento/metricas_resultados_completas.csv'))

print(f'Métricas guardadas en: ./Entrenamiento/metricas_resultados_completas.csv')
print('Historial de entrenamiento guardado en: ./Entrenamiento/historial_entrenamiento.csv')

# Leer historial completo de entrenamientos
historial_general = pd.read_csv('./Entrenamiento/historial_entrenamiento.csv')
metricas_general = pd.read_csv('./Entrenamiento/metricas_resultados_completas.csv')

# Filtrar datos para AUC
auc_data = metricas_general[metricas_general['AUC'].notna()]

# Función para guardar gráficos individuales de Accuracy y Loss
def save_individual_graph(hist_df, metric, folder_name, prefix):
    files = os.listdir(folder_name)
    num_files = len([f for f in files if f.startswith(prefix)])
    file_path = os.path.join(folder_name, f"{prefix}{num_files}.png")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x=hist_df.index, y=metric, label=f'{metric.capitalize()} Entrenamiento')
    sns.lineplot(data=hist_df, x=hist_df.index, y=f'val_{metric}', label=f'{metric.capitalize()} Validación')
    plt.title(f'{metric.capitalize()} - Entrenamiento {ID_Entrenamiento}')
    plt.xlabel('Épocas')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(file_path)
    plt.close()
    print(f"Gráfico guardado en: {file_path}")

# Guardar gráficos de Accuracy y Loss
save_individual_graph(hist_df, 'accuracy', './Graficas/Accuracy', 'comprobacion_accuracy')
save_individual_graph(hist_df, 'loss', './Graficas/Loss', 'comprobacion_loss')

# Guardar AUC en CSV acumulativo
auc_data_path = './Graficas/AUC/auc_data.csv'
new_auc = pd.DataFrame({'ID_Entrenamiento': [ID_Entrenamiento], 'AUC': [auc]})
if os.path.exists(auc_data_path):
    auc_data = pd.read_csv(auc_data_path)
    auc_data = pd.concat([auc_data, new_auc], ignore_index=True)
else:
    auc_data = new_auc
auc_data.to_csv(auc_data_path, index=False)

# Generar gráfico acumulativo de AUC
plt.figure(figsize=(10, 6))
sns.lineplot(data=auc_data, x='ID_Entrenamiento', y='AUC', marker='o', errorbar=None)
plt.title('Comparación de AUC entre Entrenamientos')
plt.xlabel('ID Entrenamiento')
plt.ylabel('AUC')
plt.savefig('./Graficas/AUC/comprobacion_auc.png')
plt.close()
print("Gráfico AUC guardado en: ./Graficas/AUC/comprobacion_auc.png")
