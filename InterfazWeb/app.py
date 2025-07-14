import os
import csv
import time
import datetime
import webbrowser
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
from Utilidades.Cargar_modelo import cargar_modelo, predecir_imagen, mejorar_imagen
from Utilidades.Grad_CAM import make_gradcam_heatmap, superponer_gradcam
from Utilidades.Generar_resultados import generar_pdf_resultado

app = Flask(__name__)
modelo = cargar_modelo('./Modelo Final/prueba_modelo_Entrenamiento_86733.keras')

# Función que genera y guarda el Grad-CAM
def generar_y_guardar_gradcam(modelo, ruta_imagen, ruta_guardado, IMG_SIZE=224):
    img_array = mejorar_imagen(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    imagen_tensor = tf.convert_to_tensor(img_array)

    heatmap = make_gradcam_heatmap(imagen_tensor, modelo)
    img_original = tf.image.decode_jpeg(tf.io.read_file(ruta_imagen), channels=3)
    img_original = tf.image.resize(img_original, [IMG_SIZE, IMG_SIZE])

    cam_image = superponer_gradcam(img_original, heatmap, IMG_SIZE=IMG_SIZE)

    plt.figure(figsize=(6, 6))
    plt.imshow(cam_image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta_guardado, bbox_inches='tight', pad_inches=0.1)
    plt.close()

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return 'No se ha subido ninguna imagen', 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return 'Nombre de archivo vacío', 400

    # Validación de extensión permitida
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not allowed_file(archivo.filename):
        return 'Formato de archivo no permitido. Solo se aceptan archivos .png, .jpg o .jpeg', 400

    ahora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"imagen_{ahora}.jpg"

    carpeta_analizadas = os.path.join(app.root_path, 'static', 'ImagenesAnalizadas')
    os.makedirs(carpeta_analizadas, exist_ok=True)
    ruta_absoluta = os.path.join(carpeta_analizadas, nombre_archivo)
    archivo.save(ruta_absoluta)

    for _ in range(5):
        if os.path.exists(ruta_absoluta) and os.path.getsize(ruta_absoluta) > 0:
            break
        time.sleep(0.4)
    else:
        return f'Error: No se pudo guardar correctamente la imagen en {ruta_absoluta}', 500

    try:
        clase_predicha, confianza = predecir_imagen(modelo, ruta_absoluta)
        ruta_original_relativa = url_for('static', filename=f'ImagenesAnalizadas/{nombre_archivo}')

        carpeta_gradcam = os.path.join(app.root_path, 'static', 'GradCAM')
        os.makedirs(carpeta_gradcam, exist_ok=True)
        ruta_gradcam = os.path.join(carpeta_gradcam, f"gradcam_{ahora}.png")

        generar_y_guardar_gradcam(modelo, ruta_absoluta, ruta_gradcam)
        ruta_gradcam_relativa = url_for('static', filename=f'GradCAM/gradcam_{ahora}.png')

        historial_path = os.path.join(app.root_path, 'static', 'historial.csv')
        nuevo_registro = [ahora, nombre_archivo, clase_predicha, f"{confianza:.2f}",
                          f'ImagenesAnalizadas/{nombre_archivo}', f'GradCAM/gradcam_{ahora}.png']
        
        archivo_existe = os.path.isfile(historial_path)
        with open(historial_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not archivo_existe:
                writer.writerow(['fecha', 'archivo', 'prediccion', 'confianza', 'ruta_original', 'ruta_gradcam'])
            writer.writerow(nuevo_registro)

        pdf_nombre = f"informe_{ahora}"
        generar_pdf_resultado(
            nombre_pdf=pdf_nombre,
            imagen_path=ruta_gradcam,
            clase_predicha=clase_predicha,
            probabilidad=confianza
        )
        ruta_pdf_relativa = url_for('static', filename=f'Resultados/{pdf_nombre}.pdf')

    except Exception as e:
        return f"Ocurrió un error al analizar la imagen: {str(e)}", 500

    return render_template('resultados.html',
                           clase=clase_predicha,
                           confianza=confianza,
                           ruta_original=ruta_original_relativa,
                           ruta_gradcam=ruta_gradcam_relativa,
                           ruta_pdf=ruta_pdf_relativa)

@app.route('/historial')
def historial():
    historial_path = os.path.join(app.root_path, 'static', 'historial.csv')
    registros = []

    if os.path.exists(historial_path):
        with open(historial_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            registros = list(reader)

    return render_template('historial.html', registros=registros)

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True)
