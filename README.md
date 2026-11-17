# Lector de partituras V5.2 — cabezas negras y blancas

Aplicación web de Streamlit. Puedes subir una fotografía o captura de una partitura para clarinete, hacer clic sobre las cabezas de las notas y obtener una imagen con sus nombres (DO RE MI FA SOL LA SI) y una lista por compás.

## Mejoras de V5.2

- **Notas negras:** mantiene el detector probado en V5.1 para óvalos rellenos.
- **Notas blancas:** incorpora un detector del interior blanco rodeado por el contorno de tinta.
- **Notas blancas sobre las líneas:** si el pentagrama divide el interior en dos huecos, combina ambos para estimar mejor el centro musical. Esto corrige varios casos en que FA se confundía con SOL.
- **Clic tolerante:** la identificación no depende de pulsar exactamente en el centro, sino de la figura cercana.
- **Controles que permanecen:** opción de desactivar el ajuste del clic, corrección de tabla, marcadores de compás y exportación a PNG o CSV.

## Subir a GitHub / Streamlit Cloud

1. Abre el repositorio donde ya tienes el lector funcionando.
2. Reemplaza `app.py`, `core.py`, `requirements.txt` y `README.md` por los de esta carpeta.
3. Confirma con **Commit changes**. Streamlit Cloud actualizará la aplicación.
4. Abre la página y comprueba que en el título se muestra **V5.2**.

## Prueba recomendada

Abre **Himno a la humildad** y marca una blanca, pulsando ligeramente por encima y por debajo de su cabeza. Repite con una blanca atravesada por una línea del pentagrama y después con una negra. La lectura asistida no añade notas automáticamente: debes indicarlas con clic.

## Limitaciones

La detección geométrica sigue siendo experimental y depende de la nitidez de la imagen. Puede fallar con trazos desvaídos, manchas o partituras manuscritas. En esos casos, desactiva el ajuste del clic y corrige el nombre desde la tabla. Las armaduras se aplican mediante controles, pero las alteraciones locales no se leen automáticamente.

## Ejecutar localmente (opcional)

```sh
pip install -r requirements.txt
streamlit run app.py
```

No necesitas instalar nada en tu computadora si usas el repositorio en Streamlit Cloud.
