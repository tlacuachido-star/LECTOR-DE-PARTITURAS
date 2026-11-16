import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

NOTES = ['DO', 'RE', 'MI', 'FA', 'SOL', 'LA', 'SI']
STEPS = ['MI', 'FA', 'SOL', 'LA', 'SI', 'DO', 'RE']  # bottom line E4 upward by staff steps
COLS = ['sistema', 'compas_estimado', 'x', 'y', 'nota', 'octava', 'alteracion', 'confianza']


def clusters(values, gap=2):
    groups = []
    for value in values:
        value = int(value)
        if not groups or value - groups[-1][-1] > gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [int(np.median(g)) for g in groups]


def find_staffs(gray):
    h, w = gray.shape
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 11), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    coverage = np.count_nonzero(horizontal, axis=1)
    ys = clusters(np.flatnonzero(coverage > w * 0.28), max(2, h // 1000))
    staffs = []
    for i in range(len(ys) - 4):
        five = ys[i:i + 5]
        ds = np.diff(five)
        d = float(np.median(ds))
        if 6 <= d <= 65 and np.max(np.abs(ds - d)) <= max(2.5, d * 0.22):
            if not staffs or five[0] > staffs[-1][-1] + 2 * d:
                staffs.append(five)
    return binary, horizontal, staffs


def pitch_from_y(y, staff):
    d = float(np.median(np.diff(staff)))
    step = int(round((staff[-1] - y) / (d / 2.0)))
    idx = step % 7
    octave = 4 + (step + 2) // 7
    return STEPS[idx], octave, step


def extent(horizontal, staff):
    h, w = horizontal.shape
    mask = np.any(horizontal[max(0, staff[0] - 2):min(h, staff[-1] + 3)] > 0, axis=0)
    runs = []
    for x in np.flatnonzero(mask):
        if not runs or x - runs[-1][-1] > 5:
            runs.append([int(x)])
        else:
            runs[-1].append(int(x))
    if not runs:
        return 0, w - 1
    main = max(runs, key=len)
    return min(main), max(main)


def build_notehead_templates(d):
    tw = max(7, int(round(0.95 * d)))
    th = max(5, int(round(0.72 * d)))
    filled = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(filled, (tw // 2, th // 2), (max(1, tw // 2 - 1), max(1, th // 2 - 2)), 0, 0, 360, 255, -1)

    hollow = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(hollow, (tw // 2, th // 2), (max(1, tw // 2 - 1), max(1, th // 2 - 2)), 0, 0, 360, 255, 1)
    cv2.dilate(hollow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), hollow)

    return [(filled, 0.0), (hollow, 0.05)]


def local_maxima(mask, values, radius_x, radius_y):
    points = []
    ys, xs = np.where(mask)
    order = np.argsort(values[ys, xs])[::-1]
    for idx in order:
        x = int(xs[idx])
        y = int(ys[idx])
        score = float(values[y, x])
        if any(abs(x - px) <= radius_x and abs(y - py) <= radius_y for px, py, _ in points):
            continue
        points.append((x, y, score))
    return points


def detect_barlines(binary, staff, start_x, right_x, d):
    h, w = binary.shape
    region = binary[max(0, int(staff[0] - 1)):min(h, int(staff[-1] + 2))]
    full = np.count_nonzero(region, axis=0)
    xs = np.flatnonzero(full > 0.91 * region.shape[0])
    bars = clusters(xs, max(1, int(d * 0.15)))
    return [b for b in bars if start_x < b < right_x - d]


def detect_notes(gray, binary, horizontal, staffs, sensitivity):
    h, w = gray.shape
    out = []

    for sn, staff in enumerate(staffs, 1):
        d = float(np.median(np.diff(staff)))
        left, right = extent(horizontal, staff)

        # Exclude clef, key signature, tempo/time signature, and most leading symbols.
        start_x = int(left + (7.8 if sn == 1 else 3.6) * d)
        end_x = int(right - 0.2 * d)
        top = max(0, int(staff[0] - 2.2 * d))
        bottom = min(h, int(staff[-1] + 2.6 * d))

        roi_binary = binary[top:bottom, start_x:end_x]
        if roi_binary.size == 0:
            continue

        templates = build_notehead_templates(d)
        score_map = np.zeros(roi_binary.shape, dtype=np.float32)

        # Match both filled and hollow notehead templates.
        for templ, bonus in templates:
            th, tw = templ.shape
            if roi_binary.shape[0] < th or roi_binary.shape[1] < tw:
                continue
            match = cv2.matchTemplate(roi_binary, templ, cv2.TM_CCOEFF_NORMED)
            padded = np.zeros_like(score_map)
            oy = th // 2
            ox = tw // 2
            padded[oy:oy + match.shape[0], ox:ox + match.shape[1]] = match + bonus
            score_map = np.maximum(score_map, padded)

        thr = 0.34 + (65 - sensitivity) * 0.003
        thr = max(0.25, min(0.50, thr))
        maxima_mask = score_map >= thr
        candidates = local_maxima(maxima_mask, score_map, max(3, int(0.7 * d)), max(3, int(0.65 * d)))

        bars = detect_barlines(binary, staff, start_x, end_x, d)

        for x0, y0, corr in candidates:
            cx = x0 + start_x
            cy = y0 + top

            # Must lie near a valid staff step according to the scale reference.
            name, octave, step = pitch_from_y(cy, staff)
            step_float = (staff[-1] - cy) / (d / 2.0)
            grid_err = abs(step_float - round(step_float))
            if grid_err > 0.34:
                continue

            # Check there is a dark oval core near the center.
            rx = max(2, int(0.48 * d))
            ry = max(2, int(0.34 * d))
            patch = binary[max(0, int(cy) - ry):min(h, int(cy) + ry + 1), max(0, int(cx) - rx):min(w, int(cx) + rx + 1)]
            density = np.count_nonzero(patch) / max(1, patch.size)
            if density < 0.16:
                continue

            # Reject candidates sitting too close to barlines or mostly vertical symbols.
            if any(abs(cx - b) < 0.35 * d for b in bars):
                continue
            tall_patch = binary[max(0, int(cy) - int(1.2 * d)):min(h, int(cy) + int(1.2 * d) + 1), max(0, int(cx) - 1):min(w, int(cx) + 2)]
            vertical_ratio = np.count_nonzero(tall_patch) / max(1, tall_patch.size)

            score = 0.55 * corr + 0.25 * min(density / 0.5, 1.0) + 0.20 * (1.0 - min(grid_err / 0.34, 1.0))
            score -= 0.08 if vertical_ratio > 0.68 else 0.0
            if score < 0.30 + (100 - sensitivity) * 0.0015:
                continue

            # Final deduplication is done later, but reject extreme outliers in x/y.
            out.append([sn, 1 + sum(b < cx for b in bars), int(round(cx)), int(round(cy)), name, octave, '', round(float(score), 2)])

    if not out:
        return pd.DataFrame(columns=COLS)

    df = pd.DataFrame(out, columns=COLS)

    # Strong deduplication inside each system using musical spacing.
    dedup_rows = []
    for sn, sub in df.sort_values('confianza', ascending=False).groupby('sistema'):
        staff = staffs[sn - 1]
        d = float(np.median(np.diff(staff)))
        kept = []
        for _, row in sub.iterrows():
            if any(abs(row.x - k.x) < 0.72 * d and abs(row.y - k.y) < 0.70 * d for k in kept):
                continue
            kept.append(row)
        dedup_rows.extend(kept)

    final = pd.DataFrame(dedup_rows, columns=COLS).sort_values(['sistema', 'x', 'y']).reset_index(drop=True)
    return final


def annotate(rgb, df):
    im = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(12, min(22, im.width // 90)))
    except OSError:
        font = ImageFont.load_default()

    for _, row in df.iterrows():
        try:
            x, y = int(row.x), int(row.y)
            if not (0 <= x < im.width and 0 <= y < im.height):
                continue
            label = str(row.nota) + str(row.alteracion if pd.notna(row.alteracion) else '')
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tx = max(0, x - tw // 2)
            ty = max(0, y - th - 22)
            draw.rectangle((tx - 2, ty - 2, tx + tw + 2, ty + th + 2), fill='white')
            draw.text((tx, ty), label, font=font, fill='#1354aa')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline='#e33b30', width=2)
        except Exception:
            pass
    return im


st.set_page_config(page_title='Lector de partituras V4', layout='wide')
st.title('🎼 Lector de partituras — DO RE MI · V4')
st.caption('Versión ajustada con referencia de clave de Sol: mejor lectura de alturas y menos falsos positivos.')

sensitivity = st.slider('Sensibilidad', 20, 100, 65, 5)
file = st.file_uploader('Sube JPG o PNG', type=['jpg', 'jpeg', 'png'])

if file:
    rgb = np.asarray(Image.open(file).convert('RGB'))
    if max(rgb.shape[:2]) > 2600:
        scale = 2600 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary, horizontal, staffs = find_staffs(gray)

    st.write('Pentagramas detectados:', len(staffs))
    if not staffs:
        st.error('No se detectaron pentagramas. Prueba una imagen más recta y nítida.')
        st.stop()

    df = detect_notes(gray, binary, horizontal, staffs, sensitivity)
    st.write('Candidatos detectados:', len(df))

    threshold = st.slider('Mostrar detecciones desde confianza', 0.0, 1.0, 0.0, 0.05)
    if threshold > 0:
        df = df[df.confianza >= threshold].copy()

    st.subheader('1. Revisa y corrige las notas')
    st.info('La altura se calcula con la escala de clave de Sol. Las alteraciones y compases aún son aproximados.')
    edited = st.data_editor(
        df,
        num_rows='dynamic',
        hide_index=True,
        use_container_width=True,
        column_config={
            'nota': st.column_config.SelectboxColumn('Nota', options=NOTES),
            'alteracion': st.column_config.SelectboxColumn('Alteración', options=['', '♯', '♭', '♮']),
        },
        key=f'edit_{file.name}_{sensitivity}_{threshold}'
    )

    st.subheader('2. Imagen anotada')
    result = annotate(rgb, edited)
    st.image(result, use_container_width=True)
    buff = io.BytesIO()
    result.save(buff, format='PNG')
    st.download_button('Descargar PNG', buff.getvalue(), 'partitura_v4.png', 'image/png')

    st.subheader('3. Notas por compás')
    if len(edited):
        safe = edited.dropna(subset=['sistema', 'compas_estimado', 'x', 'nota']).copy()
        safe['texto'] = safe.nota.astype(str) + safe.alteracion.fillna('').astype(str)
        grouped = (
            safe.sort_values(['sistema', 'compas_estimado', 'x'])
            .groupby(['sistema', 'compas_estimado'], sort=True)['texto']
            .apply(lambda v: ' – '.join(v))
            .reset_index()
        )
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.download_button('Descargar CSV', grouped.to_csv(index=False).encode('utf-8-sig'), 'notas_v4.csv', 'text/csv')
