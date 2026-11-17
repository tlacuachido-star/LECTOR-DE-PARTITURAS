# Lector de partituras V5 — modo asistido

Corrección de la V4: **el reconocimiento automático no es fiable** y se deja como sugerencia opcional. El modo predeterminado es asistido: haces clic en la cabeza de la nota y el programa calcula su altura usando las cinco líneas del pentagrama.

## Cómo usar

1. Carga JPG o PNG.
2. Selecciona armadura (por ejemplo 1 sostenido = FA♯).
3. Selecciona pentagrama y mitad izquierda o derecha para ver las notas más grandes.
4. Haz clic sobre **el centro de cada cabeza de nota**. Aparecerá el nombre.
5. Para separar compases cambia la acción a **Marcar barra** y haz clic sobre las barras verticales.
6. Para borrar una nota equivocada usa **Borrar nota cercana** o *Deshacer última nota*.
7. Corrige nombre y alteración en la tabla y descarga PNG o CSV.

## Avisos

- La detección automática heredada de V3 es experimental y no se ejecuta sola.
- La lectura asistida identifica la altura según la clave de Sol (MI en la línea inferior). El nombre mostrado es el **escrito** para el instrumento, sin transponer a tono de concierto.
- La armadura se aplica por selección del usuario; no se detectan los accidentales locales automáticamente.
- Los números de compás son **locales a cada pentagrama** y requieren marcar sus barras para separar notas.
- No hay que instalar nada en Windows si se ejecuta en Streamlit Cloud.

## Despliegue

Sube `app.py`, `core.py`, `requirements.txt` y `README.md` a la raíz del repositorio de GitHub y Streamlit recargará la app automáticamente.
