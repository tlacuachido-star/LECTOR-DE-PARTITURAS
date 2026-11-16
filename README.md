# Lector de partituras V3

Detector experimental de notas escritas en clave de Sol. La V3 recupera una búsqueda amplia de cabezas de nota y calcula su altura usando las cinco líneas de cada pentagrama. Permite corrección manual, imagen anotada y CSV por compases aproximados.

## Publicar en Streamlit
Reemplaza `app.py`, `requirements.txt` y `README.md` en la rama `main` de GitHub. Streamlit Cloud redeplegará automáticamente.

## Advertencias
No es reconocimiento musical entrenado ni validado. Puede omitir notas o marcar símbolos falsos. No interpreta armaduras, alteraciones ni compases de forma confiable. La confianza es una heurística, no una probabilidad calibrada. Revisa los resultados antes de tocar.
