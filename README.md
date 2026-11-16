# Lector experimental de partituras — V4

Aplicación en Streamlit para subir una imagen de partitura, detectar notas aproximadas y producir:

- una imagen anotada con nombres DO, RE, MI, FA, SOL, LA, SI;
- una tabla editable de notas;
- una lista agrupada por compás;
- exportación a PNG y CSV.

## Cambios de V4

- Ajuste de alturas usando la referencia de **clave de Sol**.
- Exclusión más fuerte de la zona de clave, armadura y compás al inicio del sistema.
- Detección por coincidencia de **cabezas de nota** (óvalos llenos y huecos).
- Menos falsos positivos en textos y símbolos verticales.
- Sensibilidad inicial en **65%**.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota

Sigue siendo una herramienta experimental. Las armaduras, alteraciones, ligaduras y compases son aproximados y deben revisarse manualmente antes de usar la transcripción.
