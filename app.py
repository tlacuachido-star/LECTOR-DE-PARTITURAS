"""Lector V5.3: reconocimiento opcional con revisión y clic asistido."""
import hashlib
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from core import (
    detectar_pentagramas, posicion_a_nota, alteracion_armadura,
    recorte_sistema, dibujar_anotaciones, compas_local, detectar_propuestas_v53, centro_cabeza_cercana
)

st.set_page_config(page_title='Lector de partituras V5.3', layout='wide')
st.title('🎼 Lector de partituras — DO RE MI · V5.3')
st.caption('Detección automática opcional, con sugerencias dudosas visibles y confirmación manual por clic.')

f = st.file_uploader('1. Sube una partitura JPG o PNG', type=['jpg','jpeg','png'])
if f is None:
    st.info('El modo asistido evita interpretar números, claves y silencios como notas. Nada se instala en tu PC.')
    st.stop()

contenido = f.getvalue()
identificador = hashlib.sha1(contenido).hexdigest()[:12]
if st.session_state.get('archivo_actual') != identificador:
    st.session_state.archivo_actual = identificador
    st.session_state.notas_v5 = []
    st.session_state.barras_v5 = []
    st.session_state.clics_procesados_v5 = set()
    st.session_state.editor_version_v5 = 0
    st.session_state.aviso_clic_v5_2 = None

rgb = np.asarray(Image.open(io.BytesIO(contenido)).convert('RGB'))
if max(rgb.shape[:2]) > 2800:
    escala = 2800/max(rgb.shape[:2])
    rgb = cv2.resize(rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

pentagramas, tinta, horizontal = detectar_pentagramas(rgb)
if not pentagramas:
    st.error('No se detectaron pentagramas rectos. Usa un recorte claro de una página de partitura.')
    st.stop()

st.success(f'Encontré {len(pentagramas)} pentagramas. La detección automática NO se añade por defecto.')

col_a, col_b = st.columns(2)
with col_a:
    tipo = st.selectbox('Armadura', ['Sostenidos', 'Sin alteraciones', 'Bemoles'])
with col_b:
    cantidad = st.selectbox('Cantidad', list(range(8)), index=1, disabled=(tipo == 'Sin alteraciones'))
if tipo == 'Sin alteraciones':
    cantidad = 0

st.subheader('2. Reconocimiento automático (opcional)')
st.caption('La detección produce SUGERENCIAS, no notas verificadas. Las blancas y figuras difíciles se marcan como dudosas.')
sensibilidad = st.slider('Sensibilidad del detector', 20, 100, 65, 5,
                         help='Cambia la cantidad de propuestas. No equivale a probabilidad de acierto.')
ca, cb = st.columns(2)
with ca:
    detectar_ahora = st.button('🔍 Detectar notas automáticamente', type='primary', use_container_width=True)
with cb:
    borrar_pendientes = st.button('Eliminar propuestas sin confirmar', use_container_width=True)

if detectar_ahora:
    detectadas = detectar_propuestas_v53(rgb, pentagramas, tinta, horizontal, sensibilidad)
    # Preserve any note that the user has confirmed or added manually.
    conservar = [n for n in st.session_state.notas_v5
                if n.get('revision', 'Confirmada' if n.get('origen')=='manual' else 'Dudosa') == 'Confirmada']
    agregadas = []
    for n in detectadas:
        d = pentagramas[n['sistema'] - 1]['distancia']
        if not any(a['sistema']==n['sistema'] and abs(a['x']-n['x']) < .9*d
                   for a in conservar + agregadas):
            agregadas.append(n)
    st.session_state.notas_v5 = conservar + agregadas
    st.session_state.editor_version_v5 += 1
    st.rerun()
if borrar_pendientes:
    st.session_state.notas_v5 = [n for n in st.session_state.notas_v5
                                 if n.get('revision','Confirmada' if n.get('origen')=='manual' else 'Dudosa') == 'Confirmada']
    st.session_state.editor_version_v5 += 1
    st.rerun()

confirmadas = sum(n.get('revision', 'Confirmada' if n.get('origen')=='manual' else 'Dudosa') == 'Confirmada'
                  for n in st.session_state.notas_v5)
probables = sum(n.get('revision') == 'Probable' for n in st.session_state.notas_v5)
dudosas = sum(n.get('revision') == 'Dudosa' for n in st.session_state.notas_v5)
x1,x2,x3 = st.columns(3)
x1.metric('🟢 Confirmadas', confirmadas)
x2.metric('🔵 Probables (sin verificar)', probables)
x3.metric('🟠 Dudosas (revisar)', dudosas)
st.caption('🟢 Verde = confirmada por ti · 🔵 Azul = propuesta aparentemente clara · 🟠 Naranja y ? = dudosa. Ninguna propuesta automática se considera confirmada.')

st.subheader('3. Pulsa sobre la cabeza de cada nota')
if st.session_state.get('aviso_clic_v5_2'):
    st.warning(st.session_state.aviso_clic_v5_2)
    st.session_state.aviso_clic_v5_2 = None
st.caption('Haz clic en la nota para añadirla o confirmar una propuesta automática. Si una blanca falla, puedes desactivar el ajuste y corregir su nombre en la tabla.')
modo_clic = st.toggle('Ajustar clic al centro de la cabeza de nota (recomendado)', value=True, help='Busca el contorno de las blancas y la masa de tinta de las negras. Si no encuentra un óvalo cercano, no añade ninguna nota.')
c1,c2,c3=st.columns([1,1,1])
with c1:
    sistema = st.selectbox('Pentagrama',list(range(1,len(pentagramas)+1)),format_func=lambda i:f'Pentagrama {i}')
with c2:
    mitad = st.radio('Zona', ['Izquierda','Derecha','Todo'],horizontal=True)
with c3:
    accion = st.radio('Acción al hacer clic',['Añadir nota','Confirmar propuesta cercana','Borrar nota cercana','Marcar barra','Borrar barra'],horizontal=False)

recuadro, imagen_base = recorte_sistema(rgb,pentagramas,sistema,mitad)
notas_en_sistema = [n for n in st.session_state.notas_v5 if n['sistema']==sistema]
barras_en_sistema = [b for b in st.session_state.barras_v5 if b['sistema']==sistema]
# Draw notes on an image at original pixel scale; enlarge after rendering.
escala_vista = 1.65 if mitad != 'Todo' else 1.0
anotada = dibujar_anotaciones(np.asarray(imagen_base), notas_en_sistema,
                              barras_en_sistema,tipo,cantidad,coordenadas=recuadro,escala=escala_vista)

respuesta = streamlit_image_coordinates(anotada, key=f'clicks-v5-3-{identificador}-{sistema}-{mitad}')
if respuesta is not None:
    # Component reports the coordinates and visible dimensions of the displayed image.
    rx = respuesta.get('x')
    ry = respuesta.get('y')
    rw = respuesta.get('width',anotada.width)
    rh = respuesta.get('height',anotada.height)
    if rx is not None and ry is not None:
        # Never process an old click again when switching action/system/half.
        xy_fingerprint = (identificador,sistema,mitad,rx,ry,respuesta.get('unix_time',respuesta.get('time')))
        if xy_fingerprint not in st.session_state.clics_procesados_v5:
            st.session_state.clics_procesados_v5.add(xy_fingerprint)
            orig_x = int(recuadro[0]+rx*imagen_base.width/max(1,rw))
            orig_y = int(recuadro[1]+ry*imagen_base.height/max(1,rh))
            pent = pentagramas[sistema-1]
            if accion=='Añadir nota':
                if modo_clic:
                    ajustada = centro_cabeza_cercana(rgb, orig_x, orig_y, pent)
                    if ajustada is None:
                        st.session_state.aviso_clic_v5_2 = 'No encontré una cabeza de nota cerca del clic. Toca más cerca del óvalo o desactiva el ajuste.'
                    else:
                        orig_x, orig_y = ajustada
                if not modo_clic or ajustada is not None:
                    n,octava = posicion_a_nota(orig_y,pent)
                else:
                    n, octava = None, None
                d = pent['distancia']
                repetida = any(a['sistema']==sistema and abs(a['x']-orig_x)<.75*d and abs(a['y']-orig_y)<.75*d
                              for a in st.session_state.notas_v5)
                if not repetida and n is not None:
                    st.session_state.notas_v5.append({'sistema':sistema,'x':orig_x,'y':orig_y,
                                                       'nota':n,'octava':octava,
                                                       'alteracion_manual':'auto','origen':'manual',
                                                       'confianza':1.0, 'revision':'Confirmada', 'tipo_cabeza':'Manual'})
            elif accion=='Confirmar propuesta cercana':
                candidatos = [a for a in st.session_state.notas_v5
                              if a['sistema'] == sistema
                              and a.get('revision', 'Dudosa') != 'Confirmada'
                              and abs(a['x']-orig_x) < 1.8*pent['distancia']
                              and abs(a['y']-orig_y) < 1.9*pent['distancia']]
                if candidatos:
                    elegido = min(candidatos,
                                  key=lambda a: (a['x']-orig_x)**2+(a['y']-orig_y)**2)
                    elegido['revision'] = 'Confirmada'
                else:
                    st.session_state.aviso_clic_v5_2 = 'No encontré una propuesta cercana para confirmar.'
            elif accion=='Borrar nota cercana':
                registros=st.session_state.notas_v5
                cercanas=[(j,abs(n['x']-orig_x)+abs(n['y']-orig_y)) for j,n in enumerate(registros)
                          if n['sistema']==sistema and abs(n['x']-orig_x)<2.4*pent['distancia']
                          and abs(n['y']-orig_y)<2.4*pent['distancia']]
                if cercanas:registros.pop(min(cercanas,key=lambda p:p[1])[0])
            elif accion=='Marcar barra':
                if not any(b['sistema']==sistema and abs(b['x']-orig_x)<pent['distancia'] for b in st.session_state.barras_v5):
                    st.session_state.barras_v5.append({'sistema':sistema,'x':orig_x,
                                                       'y0':pent['lineas'][0], 'y1':pent['lineas'][-1]})
            elif accion=='Borrar barra':
                registros=st.session_state.barras_v5
                cercanas=[(j,abs(b['x']-orig_x)) for j,b in enumerate(registros)
                          if b['sistema']==sistema and abs(b['x']-orig_x)<2.4*pent['distancia']]
                if cercanas: registros.pop(min(cercanas,key=lambda p:p[1])[0])
            st.session_state.editor_version_v5 += 1
            st.rerun()

q1,q2,q3=st.columns(3)
with q1:
    st.metric('Notas marcadas',len(st.session_state.notas_v5))
with q2:
    st.metric('Barras marcadas',len(st.session_state.barras_v5))
with q3:
    if st.button('Deshacer última nota') and st.session_state.notas_v5:
        st.session_state.notas_v5.pop()
        st.session_state.editor_version_v5 += 1
        st.rerun()

st.subheader('4. Revisar y corregir')
filas=[]
for a in sorted(st.session_state.notas_v5,key=lambda n:(n['sistema'],n['x'],n['y'])):
    f=dict(a)
    f['revision'] = f.get('revision','Confirmada' if f.get('origen')=='manual' else 'Dudosa')
    f['tipo_cabeza'] = f.get('tipo_cabeza', 'Manual' if f.get('origen')=='manual' else 'Sin clasificar')
    f['compas']=compas_local(f['x'],f['sistema'],st.session_state.barras_v5)
    f['alteracion']=alteracion_armadura(f['nota'],tipo,cantidad) if f['alteracion_manual']=='auto' else f['alteracion_manual']
    filas.append(f)

if filas:
    tabla=pd.DataFrame(filas)
    editada=st.data_editor(tabla,hide_index=True,use_container_width=True,
          column_config={
              'nota':st.column_config.SelectboxColumn('Nota',options=['DO','RE','MI','FA','SOL','LA','SI']),
              'alteracion_manual':st.column_config.SelectboxColumn('Alteración: auto o manual',options=['auto','','♯','♭','♮']),
              'origen':st.column_config.TextColumn('Origen'),
              'revision':st.column_config.SelectboxColumn('Revisión', options=['Confirmada','Probable','Dudosa'], help='Cambia a Confirmada solamente después de comprobar la nota.'),
              'tipo_cabeza':st.column_config.TextColumn('Figura estimada'),
              'compas':st.column_config.NumberColumn('Compás local'),
          },disabled=['sistema','compas','x','y','octava','alteracion','origen','confianza','tipo_cabeza'],
          key=f'tabla-v5-3-{identificador}-{st.session_state.editor_version_v5}')
    if not editada.equals(tabla):
        nuevos=[]
        for _,fila in editada.iterrows():
            if fila['nota'] in ['DO','RE','MI','FA','SOL','LA','SI']:
                nuevos.append({k:fila[k] for k in ['sistema','x','y','nota','octava','alteracion_manual','origen','confianza','revision','tipo_cabeza']})
        st.session_state.notas_v5=nuevos
        st.session_state.editor_version_v5+=1
        st.rerun()
else:
    st.info('Empieza pulsando las cabezas de nota en el pentagrama elegido.')

st.subheader('5. Descargar imagen y lista por compás')
result=dibujar_anotaciones(rgb,st.session_state.notas_v5,st.session_state.barras_v5,tipo,cantidad)
st.image(result,use_container_width=True)
buf=io.BytesIO();result.save(buf,format='PNG')
st.download_button('Descargar imagen con nombres PNG',buf.getvalue(),'partitura_con_notas_v5_3.png','image/png')
if filas:
    salida=tabla.sort_values(['sistema','compas','x']).copy()
    salida['nombre_completo']=(salida['nota']+salida['alteracion']+
                              salida['revision'].map({'Dudosa':'?', 'Probable':'~', 'Confirmada':''}).fillna('?'))
    agrupada=(salida.groupby(['sistema','compas'],sort=True)['nombre_completo']
              .apply(lambda notas:' – '.join(notas.tolist())).reset_index())
    agrupada.columns=['Pentagrama','Compás local','Notas']
    st.dataframe(agrupada,use_container_width=True,hide_index=True)
    st.download_button('Descargar lista CSV',agrupada.to_csv(index=False).encode('utf-8-sig'),
                       'notas_por_compas_v5_3.csv','text/csv')
st.caption('En el CSV: ? = dudosa, ~ = probable todavía no confirmada. Los compases son locales a cada pentagrama; marca barras si las necesitas. La armadura se aplica según el selector, pero no se leen los accidentales locales.')
