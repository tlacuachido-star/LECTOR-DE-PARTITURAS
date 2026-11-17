# Lector de partituras V5.3 — automático con revisión

Aplicación web para Streamlit Community Cloud. Subes una imagen JPG/PNG de una partitura en clave de Sol y recibes sugerencias de notas musicales en DO–RE–MI–FA–SOL–LA–SI, una imagen anotada y una lista CSV por compás.

## Novedades de V5.3

**Botón visible «Detectar notas automáticamente».** Busca cabezas de notas rellenas y contornos huecos mediante dos procedimientos independientes. Utiliza las cinco líneas detectadas para deducir la altura. Descarta zonas que parecen barras largas de silencio. No interpreta la canción por audio.

**Las sugerencias están diferenciadas por estado:**

- **Azul — Probable:** la figura detectada parece una cabeza negra de nota. No está confirmada.
- **Naranja y signo `?` — Dudosa:** detección insuficiente, especialmente notas blancas, huecos o figuras extrañas. Requiere tu revisión.
- **Verde — Confirmada:** la revisaste y confirmaste tú; no significa que el software haya comprobado por sí mismo la exactitud.

**Cómo revisar:**

1. Sube la partitura y confirma la armadura (cantidad de sostenidos o bemoles).
2. Deja la sensibilidad en **65** y haz clic en **Detectar notas automáticamente**.
3. Selecciona pentagrama y mitad de página para ampliar. Selecciona **Confirmar propuesta cercana** y pulsa sobre las propuestas que reconozcas correctamente.
4. Si no aparece una nota, utiliza **Añadir nota** con el clic asistido. Para una blanca que no encaja, desactiva el ajuste y corrige la nota en la tabla.
5. En la tabla puedes editar el nombre de cada nota, su alteración manual y su estado de revisión.
6. Descarga PNG y CSV. En CSV, el sufijo `?` significa dudosa y `~` significa probable pero pendiente de confirmar.

El botón **Eliminar propuestas sin confirmar** borra solo las propuestas azules y naranjas, preservando tus notas confirmadas.

## Instalación en GitHub

Sube o reemplaza los cuatro archivos **en la raíz** del mismo repositorio:

- `app.py`
- `core.py`
- `requirements.txt`
- `README.md`

Haz **Commit changes**. Streamlit Community Cloud desplegará la nueva versión desde GitHub. No necesitas instalar Python en tu PC.

## Limitaciones y precauciones

Este programa es un **prototipo de reconocimiento geométrico**, no un reconocedor de partituras entrenado ni validado. Puede omitir notas verdaderas e interpretar símbolos como notas; las **blancas pueden fallar más que las negras**. La puntuación llamada confianza es solo una **heurística interna**, NO un porcentaje de precisión. Un estado «Probable» no significa correcto. Las alteraciones accidentales junto a notas individuales no se detectan automáticamente y los compases requieren marcar barras en cada pentagrama. **No conviene utilizar la salida automática para interpretar música sin revisarla.**

## Ejecución local opcional

```bash
pip install -r requirements.txt
streamlit run app.py
```
