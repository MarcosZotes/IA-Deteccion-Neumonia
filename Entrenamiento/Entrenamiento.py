# Marcos Zotes Calleja
# Universidad Internacional de la Rioja
# Trabajo de Fin de Estudios, 2024-2025

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hiperparámetros
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.65  # Aumentado para mitigar sobreajuste
MODEL_SAVE_PATH = "Modelos Guardados"


def cargar_datos():
    """
    Carga los datos preprocesados guardados en los archivos npy.
    """
    X_train = np.load('./AnalisisDatos/TRAIN/X.npy')
    y_train = np.load('./AnalisisDatos/TRAIN/y.npy')

    X_val = np.load('./AnalisisDatos/VAL/X.npy')
    y_val = np.load('./AnalisisDatos/VAL/y.npy')

    X_test = np.load('./AnalisisDatos/TEST/X.npy')
    y_test = np.load('./AnalisisDatos/TEST/y.npy')

    print(f"Datos cargados: TRAIN - {X_train.shape[0]} imágenes, VAL - {X_val.shape[0]} imágenes, TEST - {X_test.shape[0]} imágenes.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model():
    """
    Construye y compila el modelo CNN con Dropout.
    """
    model = models.Sequential([

        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE),      

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1, activation='sigmoid')  # Salida binaria (NORMAL o PNEUMONIA)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def plot_history(history, save=True):
    """
    Grafica la precisión y la pérdida del entrenamiento.
    """
    plt.figure(figsize=(12, 5))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Precisión')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdida')
    plt.legend()

    if save:
        # Guardar las gráficas generadas
        os.makedirs('Graficas', exist_ok=True)
        plt.savefig(f"Graficas/historial_entrenamiento_{len(os.listdir('Graficas')) + 1}.png")
    plt.show()


def train_and_save_model():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cargar_datos()

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = build_model()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    # Entrenamiento con Data Augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=int(np.ceil(len(X_train) / BATCH_SIZE)),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler]
    )

    plot_history(history)

    # Guardar el modelo entrenado
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_save_file = os.path.join(MODEL_SAVE_PATH, "modelo_final.h5")
    model.save(model_save_file)
    print(f"\n✅ Modelo guardado en: {model_save_file}")


if __name__ == "__main__":
    train_and_save_model()
