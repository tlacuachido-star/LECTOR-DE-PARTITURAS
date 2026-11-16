# Lector experimental de partituras

Aplicación local en Python/Streamlit para subir una imagen de una partitura, detectar pentagramas en clave de Sol y producir una imagen anotada y una lista de notas por compás.

## Instalación

Requiere Python 3.10 o posterior.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una dirección local en el navegador. Sube un JPG o PNG, corrige la tabla y descarga los resultados.

## Limitaciones importantes

**Prototipo experimental, no una transcripción confiable sin revisión.** El detector geométrico está diseñado para notas negras impresas de tamaño regular. Puede perder notas en líneas, blancas, acordes y grupos de corcheas; puede confundir símbolos con notas. La clave de Sol se presupone. No reconoce automáticamente sostenidos, bemoles, becuadros ni la armadura: edita la columna `alteracion`. La separación en compases es estimada. No transpone de Si♭ a tono de concierto; muestra los nombres escritos.

Para alcanzar precisión profesional sería necesario integrar un motor OMR (por ejemplo Audiveris), interpretar MusicXML y validar con partituras de referencia etiquetadas. No se ha entrenado ni validado un modelo propio.
