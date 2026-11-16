"""Lector experimental de notas para partituras impresas en clave de sol."""
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

NAMES = ['DO', 'RE', 'MI', 'FA', 'SOL', 'LA', 'SI']
# Positions from bottom line E4: 0 E, 1 F, 2 G, etc.
DIATONIC = ['MI', 'FA', 'SOL', 'LA', 'SI', 'DO', 'RE']


def groups(values, tolerance=2):
    result = []
    for v in values:
        if not result or v - result[-1][-1] > tolerance:
            result.append([int(v)])
        else:
            result[-1].append(int(v))
    return [int(np.median(x)) for x in result]


def detect_staffs(gray):
    h, w = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 9), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    coverage = np.count_nonzero(horizontal, axis=1)
    ys = groups(np.flatnonzero(coverage > w * 0.30), 2)
    staffs = []
    for i in range(len(ys) - 4):
        five = ys[i:i + 5]
        d = np.diff(five)
        if 5 <= np.median(d) <= 50 and max(abs(d - np.median(d))) <= max(2.5, np.median(d) * .20):
            if not staffs or five[0] > staffs[-1][-1] + np.median(d) * 2:
                staffs.append(five)
    return binary, horizontal, staffs


def pitch_at(y, staff):
    spacing = float(np.median(np.diff(staff)))
    step = int(round((staff[4] - y) / (spacing / 2)))
    name = DIATONIC[step % 7]
    # E4 index 0; each seven diatonic steps advances octave
    octave = 4 + (step + 2) // 7
    return name, octave, step


def find_barlines(binary, staff):
    h, w = binary.shape
    spacing = int(np.median(np.diff(staff)))
    y1, y2 = max(0, staff[0] - 1), min(h, staff[4] + 2)
    crop = binary[y1:y2]
    counts = np.count_nonzero(crop, axis=0)
    xs = groups(np.flatnonzero(counts >= (y2-y1) * .83), 2)
    return [x for x in xs if 0 < x < w-1]


def detect_notes(gray, binary, horizontal, staffs):
    h, w = gray.shape
    # Remove horizontal staff lines, leaving noteheads, stems, and other marks.
    removed = cv2.subtract(binary, horizontal)
    notes = []
    for system, staff in enumerate(staffs, 1):
        spacing = float(np.median(np.diff(staff)))
        top = max(0, int(staff[0] - 4 * spacing))
        bottom = min(h, int(staff[4] + 4 * spacing))
        # Horizontal filled ovals; this is intentionally conservative.
        kw = max(3, int(round(spacing * .75)))
        kh = max(2, int(round(spacing * .26)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
        roi = removed[top:bottom]
        blobs = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(blobs)
        bars = find_barlines(binary, staff)
        candidates = []
        for k in range(1, count):
            x, y, bw, bh, area = stats[k]
            cx, cy = centers[k]
            cy += top
            if not (spacing * .45 <= bw <= spacing * 2.1):
                continue
            if not (spacing * .16 <= bh <= spacing * 1.15):
                continue
            if area < max(4, spacing * spacing * .075):
                continue
            if any(abs(cx - b) < spacing * .35 for b in bars):
                continue
            name, octave, step = pitch_at(cy, staff)
            candidates.append((float(cx), float(cy), name, octave, step))
        candidates.sort(key=lambda v: v[0])
        # Suppress overlapping candidates from same notehead.
        unique = []
        for item in candidates:
            if any(abs(item[0] - old[0]) < spacing * .65 and abs(item[1] - old[1]) < spacing * .55 for old in unique):
                continue
            unique.append(item)
        for cx, cy, name, octave, step in unique:
            # Bar numbering is approximate: clefs, repeat signs and multiple systems
            # can cause false barline detections.
            bar = 1 + sum(b < cx for b in bars)
            notes.append(dict(sistema=system, compas_estimado=bar, x=round(cx), y=round(cy),
                              nota=name, octava=octave, alteracion='', confianza='Revisar'))
    return pd.DataFrame(notes, columns=['sistema','compas_estimado','x','y','nota','octava','alteracion','confianza'])


def draw_labels(rgb, df):
    img = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    for _, row in df.iterrows():
        x, y = int(row.x), int(row.y)
        label = f"{row.nota}{row.alteracion}"
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        px, py = max(0, x - tw // 2), max(0, y - th - 22)
        draw.rectangle((px-2, py-1, px+tw+2, py+th+2), fill='white')
        draw.text((px, py), label, fill=(0, 45, 170), font=font)
        draw.ellipse((x-3, y-3, x+3, y+3), outline=(220, 0, 0), width=2)
    return img


st.set_page_config(page_title='Lector de partituras', layout='wide')
st.title('🎼 Lector de partituras — DO RE MI')
st.write('Prototipo para **clave de Sol**. Detecta notas aproximadas y permite corregirlas antes de exportar. **No es un lector musical validado**.')
file = st.file_uploader('Sube una imagen JPG o PNG', type=['jpg', 'jpeg', 'png'])
if file:
    source = Image.open(file).convert('RGB')
    rgb = np.array(source)
    if rgb.shape[1] > 2200:
        ratio = 2200 / rgb.shape[1]
        rgb = cv2.resize(rgb, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary, horizontal, staffs = detect_staffs(gray)
    st.write(f'Pentagramas detectados: **{len(staffs)}**')
    if not staffs:
        st.warning('No se encontraron pentagramas. Prueba una imagen frontal, nítida y sin inclinación.')
        st.image(rgb)
        st.stop()
    df = detect_notes(gray, binary, horizontal, staffs)
    st.warning('IMPORTANTE: las detecciones pueden incluir símbolos falsos y omitir notas, especialmente blancas, corcheas unidas y notas en líneas. Comprueba contra la partitura original. Las alteraciones no se detectan automáticamente.')
    st.subheader('1. Corregir las notas')
    st.caption('Puedes editar, añadir o eliminar filas. El número de compás es estimado; corrígelo si hace falta. Para armadura de un sostenido, marca FA♯ donde corresponda.')
    edited = st.data_editor(df, num_rows='dynamic', hide_index=True, use_container_width=True,
                            column_config={
                                'nota': st.column_config.SelectboxColumn('nota', options=NAMES, required=True),
                                'alteracion': st.column_config.SelectboxColumn('alteracion', options=['','♯','♭','♮'], required=False),
                                'confianza': st.column_config.TextColumn('revisión')}, key='editor')
    st.subheader('2. Imagen con nombres')
    annotated = draw_labels(rgb, edited)
    st.image(annotated, use_container_width=True)
    out = io.BytesIO()
    annotated.save(out, format='PNG')
    st.download_button('Descargar imagen PNG', out.getvalue(), 'partitura_con_nombres.png', 'image/png')
    st.subheader('3. Lista por compases')
    if len(edited):
        safe = edited.copy()
        safe['texto'] = safe['nota'].fillna('') + safe['alteracion'].fillna('')
        grouped = safe.sort_values(['sistema','compas_estimado','x']).groupby(['sistema','compas_estimado'], sort=True)['texto'].apply(lambda s:' – '.join(s)).reset_index()
        grouped.columns = ['Sistema','Compás estimado','Notas']
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.download_button('Descargar lista CSV', grouped.to_csv(index=False).encode('utf-8-sig'), 'notas_por_compas.csv', 'text/csv')
    else:
        st.info('No se detectaron notas. Puedes añadirlas manualmente en la tabla.')
