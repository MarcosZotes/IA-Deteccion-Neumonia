import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Ruta al archivo CSV generado por el comparador
ruta_csv = './Testeo/Comparativas/Metricas/comparativa_ultimos_modelos.csv'
df = pd.read_csv(ruta_csv)

# Asegurar carpeta de salida
graficas_path = './Testeo/Comparativas/Graficas'
os.makedirs(graficas_path, exist_ok=True)

# Gráficas de barras por métrica global
metricas_globales = ['Val_Accuracy', 'Macro_F1', 'AUC', 'Macro_Precision', 'Macro_Recall']
sns.set(style="whitegrid")

for metrica in metricas_globales:
    plt.figure(figsize=(12, 6))
    ordenado = df.sort_values(by=metrica, ascending=False)
    sns.barplot(x='Nombre_Modelo', y=metrica, data=ordenado, hue='Nombre_Modelo', palette='Blues_d', legend=False)
    plt.title(f'Comparativa de {metrica}', fontsize=16)
    plt.xlabel('Modelo')
    plt.ylabel(metrica)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(graficas_path, f'{metrica}_barplot.png'))
    plt.close()

print("[✔] Gráficas de barras generadas.")

# Radar plot por clase para top 5
from math import pi

top3 = df.sort_values(by='Macro_F1', ascending=False).head(5)

for _, row in top3.iterrows():
    etiquetas = ['Precision_Clase_0', 'Recall_Clase_0', 'F1_Clase_0',
                'Precision_Clase_1', 'Recall_Clase_1', 'F1_Clase_1',
                'Precision_Clase_2', 'Recall_Clase_2', 'F1_Clase_2']

    
    valores = [row[e] for e in etiquetas]
    valores += valores[:1]  # cerrar la curva

    angles = [n / float(len(etiquetas)) * 2 * pi for n in range(len(etiquetas))]
    angles += angles[:1]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], etiquetas, color='grey', size=8, rotation=45)
    ax.plot(angles, valores, linewidth=2, linestyle='solid', label=row['Nombre_Modelo'])
    ax.fill(angles, valores, alpha=0.3)
    plt.title(f'Radar por clase - {row["Nombre_Modelo"]}', size=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graficas_path, f'radar_{row["Nombre_Modelo"]}.png'))
    plt.close()

print("[✔] Radar plots por clase generados para TOP 3 modelos.")

# Scatter plot: Macro F1 vs AUC
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Macro_F1', y='AUC', hue='Arquitectura', style='Arquitectura', s=100)
for _, row in df.iterrows():
    plt.text(row['Macro_F1'] + 0.005, row['AUC'] + 0.005, row['Nombre_Modelo'], fontsize=8)
plt.title('Relación entre Macro F1 y AUC')
plt.xlabel('Macro F1')
plt.ylabel('AUC')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(graficas_path, 'scatter_f1_auc.png'))
plt.close()

print("[✔] Scatter plot F1 vs AUC generado.")

# === EXPORTAR A .md ===
md_path = os.path.join(graficas_path, 'graficas_resultados.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Comparativa de Modelos IA - Gráficas\n\n")

    # Gráficas de barras
    for metrica in metricas_globales:
        f.write(f"## {metrica.replace('_', ' ')}\n")
        f.write(f"![{metrica}](./{metrica}_barplot.png)\n\n")

    # Radar plots
    for _, row in top3.iterrows():
        f.write(f"## Radar por clase - {row['Nombre_Modelo']}\n")
        f.write(f"![Radar_{row['Nombre_Modelo']}](./radar_{row['Nombre_Modelo']}.png)\n\n")

    # Scatter plot
    f.write("## Relación entre Macro F1 y AUC\n")
    f.write("![Scatter F1 AUC](./scatter_f1_auc.png)\n\n")

print(f"[✔] Archivo Markdown generado: {md_path}")

# === EXPORTAR A .tex ===
tex_path = os.path.join(graficas_path, 'graficas_resultados.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write("\\section{Comparativa de Modelos IA - Gráficas}\n")

    # Gráficas de barras
    for metrica in metricas_globales:
        f.write(f"\\subsection{{{metrica.replace('_', ' ')}}}\n")
        f.write("\\begin{figure}[H]\n\\centering\n")
        f.write(f"\\includegraphics[width=0.9\\textwidth]{{Graficas/{metrica}_barplot.png}}\n")
        f.write(f"\\caption{{Comparativa de {metrica.replace('_', ' ')}}}\n")
        f.write("\\end{figure}\n\n")

    # Radar plots
    for _, row in top3.iterrows():
        f.write(f"\\subsection{{Radar por clase - {row['Nombre_Modelo']}}}\n")
        f.write("\\begin{figure}[H]\n\\centering\n")
        f.write(f"\\includegraphics[width=0.75\\textwidth]{{Graficas/radar_{row['Nombre_Modelo']}.png}}\n")
        f.write(f"\\caption{{Radar plot por clase del modelo {row['Nombre_Modelo']}}}\n")
        f.write("\\end{figure}\n\n")

    # Scatter plot
    f.write("\\subsection{Relación entre Macro F1 y AUC}\n")
    f.write("\\begin{figure}[H]\n\\centering\n")
    f.write("\\includegraphics[width=0.8\\textwidth]{Graficas/scatter_f1_auc.png}\n")
    f.write("\\caption{Relación entre Macro F1 y AUC para los modelos evaluados}\n")
    f.write("\\end{figure}\n")

print(f"[✔] Archivo LaTeX generado: {tex_path}")
