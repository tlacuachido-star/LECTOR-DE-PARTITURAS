# Lector de partituras — versión 2

Aplicación experimental Streamlit para leer nombres de notas en clave de Sol desde una imagen JPG/PNG. Permite corregir detecciones y exportar imagen PNG y lista CSV por compases estimados.

## Actualizar en GitHub

Reemplaza en la raíz del repositorio los archivos `app.py`, `requirements.txt` y `README.md` por los de este paquete. Confirma los cambios (Commit changes). Streamlit Community Cloud suele actualizar la aplicación automáticamente.

## Mejoras v2

- Busca cinco líneas con separación uniforme.
- Excluye la zona inicial de clave, armadura y compás.
- Filtra tamaño, forma, densidad, posición y cercanía a barras.
- Elimina candidatos superpuestos.
- Detecta posibles barras para separar compases aproximados.
- Sensibilidad predeterminada 65% y opción de confianza alta.
- Editor manual y exportaciones PNG/CSV.

## Limitaciones

No es un sistema OMR entrenado ni validado. Puede omitir notas blancas, notas sobre líneas, acordes, líneas adicionales y notas con muchas plicas; puede confundir algunos símbolos. La detección de compases es aproximada. **No aplica automáticamente sostenidos, bemoles, becuadros ni transposición para clarinete en Si♭.** Revisa la imagen y el CSV antes de tocar. Prueba imágenes nítidas y recortes de 1–2 sistemas.
