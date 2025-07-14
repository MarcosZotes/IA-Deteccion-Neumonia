# =============================================================================
# Nombre del archivo: Generar_resultados.py
# Autor: Marcos Zotes Calleja
# Proyecto: Herramienta médica para detección de neumonía basada en IA
# Descripción: Generación de informes en PDF con los resultados del modelo
# =============================================================================

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from flask import current_app
from PIL import Image
import os

def generar_pdf_resultado(nombre_pdf, imagen_path, clase_predicha, probabilidad):
    try:
        ruta_carpeta = os.path.join(current_app.root_path, 'static', 'Resultados')
        os.makedirs(ruta_carpeta, exist_ok=True)
        ruta_salida = os.path.join(ruta_carpeta, f"{nombre_pdf}.pdf")

        c = canvas.Canvas(ruta_salida, pagesize=A4)
        width, height = A4

        # === Cabecera ===
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, height - 50, "Informe de Resultados – IA Neumonía")

        # === Información básica ===
        c.setFont("Helvetica", 12)
        fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        c.drawString(50, height - 90, f"Fecha del análisis: {fecha}")
        c.drawString(50, height - 110, f"Clase predicha: {clase_predicha}")
        c.drawString(50, height - 130, f"Confianza del modelo en esta predicción: {probabilidad:.2f}%")

        # === Texto explicativo ===
        c.setFont("Helvetica", 10)
        explicacion = (
            "El modelo IA ha identificado regiones destacadas en la radiografía\n"
            "mediante un mapa de calor (Grad-CAM). Las áreas más intensas indican\n"
            "dónde se ha concentrado la atención del modelo para tomar su decisión.\n"
            "Esto permite una interpretación visual clara del resultado."
        )
        y_texto = height - 170
        for linea in explicacion.split("\n"):
            c.drawString(50, y_texto, linea)
            y_texto -= 14

        # === Imagen justo debajo del texto ===
        if os.path.exists(imagen_path):
            try:
                with Image.open(imagen_path) as img:
                    temp_path = os.path.join(ruta_carpeta, "temp_gradcam.png")
                    img.convert("RGB").save(temp_path, format="PNG")

                    ancho_img = 280
                    alto_img = 280
                    x_img = (width - ancho_img) / 2
                    y_img = y_texto - alto_img - 20  # justo debajo del texto

                    c.drawImage(temp_path, x_img, y_img, width=ancho_img, height=alto_img)
                    os.remove(temp_path)

            except Exception as e:
                print(f"[❌] Error al insertar imagen Grad-CAM: {e}")
        else:
            print(f"[❌] Imagen Grad-CAM no encontrada en: {imagen_path}")

        # === Firma final ===
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 40, "Generado automáticamente por el sistema IA – TFG UNIR.")

        c.save()

        if os.path.exists(ruta_salida):
            print(f"✅ Informe generado correctamente en: {ruta_salida}")
        else:
            print(f"[❌] Falló la creación del archivo: {ruta_salida}")

        return ruta_salida

    except Exception as e:
        print(f"[❌] Error general al generar PDF: {e}")
        return None


