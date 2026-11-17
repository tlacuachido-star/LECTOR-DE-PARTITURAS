# Lector de partituras V5.1 — clic inteligente

Permite subir una partitura en JPG/PNG a Streamlit, marcar las notas con un clic, editarlas, exportar la imagen anotada y consultar una lista por compases.

## Corrección de la V5

La V5 calculaba la nota con la coordenada `y` exacta del clic, de modo que pulsar un poco encima o debajo podía cambiar SOL por LA, por ejemplo.

Ahora, con **Ajustar clic al centro de la cabeza de nota**, un pequeño algoritmo de visión por computadora busca el óvalo negro más cercano en la imagen original. Si lo encuentra dentro del radio de tolerancia, usa su centro para calcular DO–RE–MI–FA–SOL–LA–SI. Si no lo encuentra, no añade una nota incorrecta. Puedes desactivar el ajuste para notas huecas o figuras de baja calidad que la detección no encuentre.

La precisión sigue siendo experimental y dependerá de la nitidez y del tipo de grabado. La armadura puede configurarse; otras alteraciones se corrigen manualmente en la tabla.

## Instalación en Streamlit Community Cloud

Reemplaza los cuatro archivos del repositorio raíz de GitHub:

- `app.py`
- `core.py`
- `requirements.txt`
- `README.md`

No hace falta instalar nada en el equipo local; Streamlit lo desplegará desde GitHub.
