
# VisualSD

**Visualización y procesamiento de datos sísmicos e infrasónicos**

VisualSD es una biblioteca de Python para el análisis de arrays sísmicos/infrasónicos, beamforming, y visualización interactiva de trazas. Proporciona herramientas para el cálculo de delays, alineación de señales, y representación gráfica mediante PyQtGraph.

## 🚀 Características

- **Beamforming**: Cálculo de Linear Stack y Product Beam
- **Alineación de señales**: Matriz de delays antisimétrica basada en coordenadas de estaciones
- **Visualización interactiva**: Widgets de PyQtGraph con sincronización de ejes
- **Procesamiento eficiente**: Operaciones vectorizadas con NumPy/SciPy
- **Compatibilidad ObsPy**: Integración nativa con objetos `Trace` y `Stream`

## 📦 Instalación

### Requisitos
- Python >= 3.11
- pip >= 23.0

### Instalación desde código fuente (modo desarrollo)

```bash
# Clonar o navegar al directorio del proyecto
cd VisualSD

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar en modo editable
pip install -e .

# O con dependencias de GUI (por defecto incluidas)
pip install -e .[gui]