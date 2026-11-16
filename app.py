"""Lector de partituras v2: detector geométrico experimental, no OMR validado."""
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

NOTES = ['DO', 'RE', 'MI', 'FA', 'SOL', 'LA', 'SI']
STEPS = ['MI', 'FA', 'SOL', 'LA', 'SI', 'DO', 'RE']
COLS = ['sistema', 'compas_estimado', 'x', 'y', 'nota', 'octava', 'alteracion', 'confianza']


def clusters(indices, gap=2):
    out = []
    for v in indices:
        v = int(v)
        if not out or v - out[-1][-1] > gap:
            out.append([v])
        else:
            out[-1].append(v)
    return [int(np.median(g)) for g in out]


def find_staffs(gray):
    h, w = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 11), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    counts = np.count_nonzero(horizontal, axis=1)
    ys = clusters(np.flatnonzero(counts > w * .28), max(2, h // 1000))
    staffs = []
    for i in range(len(ys) - 4):
        five = ys[i:i+5]
        dif = np.diff(five)
        d = float(np.median(dif))
        if 6 <= d <= 65 and np.max(np.abs(dif - d)) <= max(2.5, d * .22):
            if not staffs or five[0] > staffs[-1][-1] + d * 2:
                staffs.append(five)
    return binary, horizontal, staffs


def get_pitch(y, staff):
    d = float(np.median(np.diff(staff)))
    step = int(round((staff[4] - y) / (d / 2)))
    return STEPS[step % 7], 4 + (step + 2) // 7


def staff_extent(horizontal, staff):
    """Find start/end of the longest staff line to exclude margins and headings."""
    h, w = horizontal.shape
    mask = np.any(horizontal[max(0, staff[0]-2):min(h, staff[-1]+3)] > 0, axis=0)
    segments = []
    for x in np.flatnonzero(mask):
        if not segments or x > segments[-1][-1] + 5:
            segments.append([int(x)])
        else:
            segments[-1].append(int(x))
    if not segments:
        return 0, w-1
    seg = max(segments, key=len)
    return seg[0], seg[-1]


def barlines(binary, staff, left, right):
    h, w = binary.shape
    d = float(np.median(np.diff(staff)))
    y0, y1 = max(0, staff[0]-1), min(h, staff[-1]+2)
    roi = binary[y0:y1]
    # A true barline crosses most of the staff and is narrow.
    hits = np.count_nonzero(roi, axis=0) >= .90 * (y1-y0)
    xs = clusters(np.flatnonzero(hits), max(1, int(d * .15)))
    return [x for x in xs if left + 3*d < x < right - d]


def detect(gray, binary, horizontal, staffs, sensitivity, confirmed_only):
    h, w = gray.shape
    # A line-free copy; noteheads overlapping lines may need manual correction.
    linefree = cv2.subtract(binary, horizontal)
    records = []
    for number, staff in enumerate(staffs, 1):
        d = float(np.median(np.diff(staff)))
        left, right = staff_extent(horizontal, staff)
        # Skip clef/key/time signature at the beginning of each system.
        start = min(right, left + (6.5 if number == 1 else 4.5) * d)
        top = max(0, int(staff[0] - 2.7*d))
        bottom = min(h, int(staff[-1] + 2.7*d))
        if bottom <= top or right <= start:
            continue
        roi = linefree[top:bottom]
        # Morphological opening selects filled, roughly oval cores.
        kw = max(3, int(round(d * .67)) | 1)
        kh = max(3, int(round(d * .30)) | 1)
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
        n, labels, stats, centers = cv2.connectedComponentsWithStats(opened)
        bars = barlines(binary, staff, left, right)
        candidates = []
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            cx, cy = float(centers[k][0]), float(centers[k][1] + top)
            if not (start < cx < right - .5*d):
                continue
            if not (.55*d <= bw <= 1.65*d and .20*d <= bh <= 1.05*d):
                continue
            if not (.13*d*d <= area <= 1.5*d*d):
                continue
            if any(abs(cx - b) < .55*d for b in bars):
                continue
            # Pitch must align with a line or space (half-staff-step grid).
            pitch_distance = abs((staff[-1]-cy)/(d/2) - round((staff[-1]-cy)/(d/2)))
            if pitch_distance > .38:
                continue
            # Check original black pixel coverage near notehead; avoid isolated staff artifacts.
            rx, ry = max(2, int(.54*d)), max(2, int(.35*d))
            xa, xb = max(0, int(cx)-rx), min(w, int(cx)+rx+1)
            ya, yb = max(0, int(cy)-ry), min(h, int(cy)+ry+1)
            patch = binary[ya:yb, xa:xb]
            density = np.count_nonzero(patch) / max(1, patch.size)
            # Inspect nearby vertical ink for a stem. This is supporting evidence, not mandatory.
            x0, x1 = max(0, int(cx-d)), min(w, int(cx+d)+1)
            y0, y1 = max(0, int(cy-1.9*d)), min(h, int(cy+1.9*d)+1)
            around = binary[y0:y1, x0:x1]
            col = np.count_nonzero(around, axis=0) if around.size else np.array([])
            has_stem = bool(len(col) and np.max(col) > 1.35*d)
            score = min(1., .40 * min(density/.62, 1.) + .22 * min(area/(.40*d*d), 1.)
                        + .20 * (1 - pitch_distance/.5) + (.18 if has_stem else 0))
            # Higher sensitivity -> lower threshold. At 65%, conservative-ish.
            threshold = .84 - sensitivity * .0035
            if score < threshold:
                continue
            if confirmed_only and score < max(.76, threshold + .13):
                continue
            name, octave = get_pitch(cy, staff)
            candidates.append((cx, cy, score, name, octave))
        # Keep the strongest of overlapping candidate regions.
        candidates.sort(key=lambda c: c[2], reverse=True)
        unique = []
        for c in candidates:
            if not any(abs(c[0]-u[0]) < .80*d and abs(c[1]-u[1]) < .60*d for u in unique):
                unique.append(c)
        for cx, cy, score, name, octave in sorted(unique, key=lambda c: c[0]):
            bar = 1 + sum(b < cx for b in bars)
            records.append([number, bar, round(cx), round(cy), name, octave, '',
                            f'{score:.0%} - ' + ('alta' if score >= .76 else 'revisar')])
    return pd.DataFrame(records, columns=COLS)


def annotate(rgb, data):
    img = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(img)
    size = max(12, min(22, rgb.shape[1] // 90))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
    except OSError:
        font = ImageFont.load_default()
    for _, row in data.iterrows():
        try:
            x, y = int(row['x']), int(row['y'])
            if not (0 <= x < img.width and 0 <= y < img.height):
                continue
            label = str(row['nota']) + str(row.get('alteracion', '') or '')
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            tx, ty = max(0, x-tw//2), max(0, y-th-size-7)
            draw.rectangle((tx-2, ty-2, tx+tw+2, ty+th+2), fill='white')
            draw.text((tx, ty), label, fill='#0644a4', font=font)
            draw.ellipse((x-3, y-3, x+3, y+3), outline='#d22c29', width=2)
        except (ValueError, TypeError):
            continue
    return img


st.set_page_config(page_title='Lector de partituras v2', layout='wide')
st.title('🎼 Lector de partituras — DO RE MI · v2')
st.caption('Prototipo experimental para partituras impresas en clave de Sol. No es reconocimiento musical validado.')
sensitivity = st.slider('Sensibilidad de detección', 20, 100, 65, 5,
                        help='Más sensibilidad detecta más notas, pero también más símbolos falsos.')
confirmed_only = st.checkbox('Mostrar solamente notas de confianza alta', value=False)
file = st.file_uploader('Sube una partitura JPG o PNG', type=['jpg', 'jpeg', 'png'])
if file:
    source = Image.open(file).convert('RGB')
    rgb = np.asarray(source)
    if max(rgb.shape[:2]) > 2600:
        scale = 2600 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary, horizontal, staffs = find_staffs(gray)
    st.write(f'Pentagramas detectados: **{len(staffs)}**')
    if not staffs:
        st.warning('No se encontraron pentagramas. Prueba una imagen recta y nítida, recortada a la música.')
        st.image(rgb, use_container_width=True)
        st.stop()
    df = detect(gray, binary, horizontal, staffs, sensitivity, confirmed_only)
    st.info('Esta versión reduce falsos positivos, pero puede omitir notas, especialmente blancas, ligadas o con líneas adicionales. No reconoce automáticamente alteraciones ni transposición. Los compases son aproximados.')
    st.subheader('1. Revisa y corrige las notas')
    st.caption('Puedes editar, añadir o eliminar filas. Corrige compases y alteraciones antes de exportar. Los nombres son notas escritas, no sonido de concierto.')
    edited = st.data_editor(df, num_rows='dynamic', hide_index=True, use_container_width=True,
                            column_config={
                                'nota': st.column_config.SelectboxColumn('Nota', options=NOTES, required=True),
                                'alteracion': st.column_config.SelectboxColumn('Alteración', options=['', '♯', '♭', '♮']),
                                'confianza': st.column_config.TextColumn('Confianza')}, key='editor_v2')
    st.subheader('2. Imagen anotada')
    result = annotate(rgb, edited)
    st.image(result, use_container_width=True)
    out = io.BytesIO()
    result.save(out, format='PNG')
    st.download_button('Descargar imagen PNG', out.getvalue(), 'partitura_notas_v2.png', 'image/png')
    st.subheader('3. Lista por compases')
    if len(edited):
        safe = edited.dropna(subset=['nota', 'sistema', 'compas_estimado', 'x']).copy()
        safe['texto'] = safe['nota'].astype(str) + safe['alteracion'].fillna('').astype(str)
        grouped = (safe.sort_values(['sistema', 'compas_estimado', 'x'])
                   .groupby(['sistema', 'compas_estimado'], sort=True)['texto']
                   .apply(lambda v: ' – '.join(v)).reset_index())
        grouped.columns = ['Sistema', 'Compás estimado', 'Notas']
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.download_button('Descargar lista CSV', grouped.to_csv(index=False).encode('utf-8-sig'),
                           'notas_por_compas_v2.csv', 'text/csv')
    else:
        st.warning('No hay notas detectadas con estos filtros. Baja la exigencia desmarcando “confianza alta”, aumenta sensibilidad o agrega filas manualmente.')
