"""Operaciones geométricas puras: se pueden probar sin abrir Streamlit."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIATONICAS = ['MI', 'FA', 'SOL', 'LA', 'SI', 'DO', 'RE']
SOSTENIDOS = ['FA', 'DO', 'SOL', 'RE', 'LA', 'MI', 'SI']
BEMOLES = ['SI', 'MI', 'LA', 'RE', 'SOL', 'DO', 'FA']


def agrupar(valores, salto=2):
    conjuntos = []
    for v in valores:
        v = int(v)
        if not conjuntos or v - conjuntos[-1][-1] > salto:
            conjuntos.append([v])
        else:
            conjuntos[-1].append(v)
    return [int(np.median(g)) for g in conjuntos]


def detectar_pentagramas(rgb):
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alto, ancho = gris.shape
    tinta = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(
        tinta, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, ancho // 11), 1))
    )
    cobertura = np.count_nonzero(horizontal, axis=1)
    niveles = agrupar(np.flatnonzero(cobertura > ancho * 0.28), max(2, alto // 1000))
    grupos = []
    for i in range(len(niveles) - 4):
        cinco = niveles[i:i+5]
        diferencias = np.diff(cinco)
        separacion = float(np.median(diferencias))
        if not (6 <= separacion <= 80):
            continue
        if np.max(np.abs(diferencias - separacion)) > max(2.5, separacion * 0.22):
            continue
        if not grupos or cinco[0] > grupos[-1]['lineas'][-1] + 2 * separacion:
            grupos.append({'lineas': [int(x) for x in cinco], 'distancia': separacion})
    return grupos, tinta, horizontal


def posicion_a_nota(y, pentagrama):
    """Paso 0 es la línea inferior MI4; cada media separación, un grado diatónico."""
    s = pentagrama['distancia'] / 2.0
    paso = round((pentagrama['lineas'][-1] - float(y)) / s)
    nombre = DIATONICAS[paso % 7]
    octava = 4 + (paso + 2) // 7
    return nombre, int(octava)


def alteracion_armadura(nota, tipo, cantidad):
    if tipo == 'Sostenidos' and nota in SOSTENIDOS[:cantidad]:
        return '♯'
    if tipo == 'Bemoles' and nota in BEMOLES[:cantidad]:
        return '♭'
    return ''


def recorte_sistema(rgb, pentagramas, numero, mitad='Izquierda'):
    pent = pentagramas[numero - 1]
    d = pent['distancia']
    alto, ancho = rgb.shape[:2]
    y0 = max(0, int(pent['lineas'][0] - 3.6 * d))
    y1 = min(alto, int(pent['lineas'][-1] + 3.7 * d))
    if mitad == 'Izquierda':
        x0, x1 = 0, ancho // 2 + int(d * 2)
    elif mitad == 'Derecha':
        x0, x1 = ancho // 2 - int(d * 2), ancho
    else:
        x0, x1 = 0, ancho
    return (x0, y0, x1, y1), Image.fromarray(rgb[y0:y1, x0:x1])


def compas_local(x, sistema, barras):
    cortes = sorted(b['x'] for b in barras if b['sistema'] == sistema)
    return 1 + sum(x > c for c in cortes)


def dibujar_anotaciones(rgb, notas, barras, tipo, cantidad, coordenadas=None, escala=1):
    img = Image.fromarray(rgb).convert('RGB')
    dib = ImageDraw.Draw(img)
    tam = max(11, min(22, img.width // 87))
    try:
        fuente = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', tam)
    except OSError:
        fuente = ImageFont.load_default()
    for p in barras:
        if coordenadas is not None:
            x0, y0, x1, y1 = coordenadas
            if not (x0 <= p['x'] < x1):
                continue
            x = int(p['x'] - x0)
        else:
            x = int(p['x'])
        y_a, y_b = int(p.get('y0', 0)), int(p.get('y1', img.height - 1))
        if coordenadas is not None:
            y_a -= coordenadas[1]
            y_b -= coordenadas[1]
        dib.line([(x, y_a), (x, y_b)], fill='#13a48e', width=2)
    for n in notas:
        x, y = int(n['x']), int(n['y'])
        if coordenadas is not None:
            x0, y0, x1, y1 = coordenadas
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            x -= x0
            y -= y0
        alter = n.get('alteracion_manual', 'auto')
        if alter == 'auto':
            alter = alteracion_armadura(n['nota'], tipo, cantidad)
        etiqueta = f"{n['nota']}{alter}"
        dib.ellipse((x-4,y-4,x+4,y+4), outline='#e43b35', width=2)
        if n.get('origen') == 'manual':
            dib.ellipse((x-6,y-6,x+6,y+6), outline='#078c49', width=2)
        borde = dib.textbbox((0, 0), etiqueta, font=fuente)
        anchura = borde[2]-borde[0]
        alto_texto = borde[3]-borde[1]
        tx = max(0, min(img.width-anchura-4, x-anchura//2))
        ty = max(0, y - alto_texto - tam - 3)
        dib.rectangle((tx-2, ty-2, tx+anchura+2, ty+alto_texto+2), fill='white')
        dib.text((tx, ty), etiqueta, fill='#1354aa', font=fuente)
    if escala != 1:
        img = img.resize((int(img.width * escala), int(img.height * escala)), Image.Resampling.LANCZOS)
    return img


def candidatos_automaticos_v3(rgb, pentagramas, tinta, horizontal, sensibilidad=65):
    """Modo heredado; es deliberadamente opcional y NO se presenta como confirmado."""
    alto, ancho = tinta.shape
    sin_lineas = cv2.subtract(tinta, horizontal)
    salida = []
    for numero, s in enumerate(pentagramas, 1):
        lineas = s['lineas']
        d = float(s['distancia'])
        # Se consideran solo objetos alrededor del pentagrama.
        comienzo = (7.0 if numero == 1 else 3.5) * d
        zona = np.any(horizontal[max(0,lineas[0]-2):min(alto,lineas[-1]+3)]>0,axis=0)
        xs = np.flatnonzero(zona)
        if len(xs) == 0:
            continue
        izquierda, derecha = int(xs[0]), int(xs[-1])
        izquierda += int(comienzo)
        arriba, abajo = max(0,int(lineas[0]-2.4*d)), min(alto,int(lineas[-1]+2.4*d))
        roi = sin_lineas[arriba:abajo]
        total, _, rects, centroides = cv2.connectedComponentsWithStats(roi)
        posibles = []
        for k in range(1,total):
            x,y,bw,bh,area = rects[k]
            cx,cy = float(centroides[k][0]),float(centroides[k][1]+arriba)
            if not (izquierda < cx < derecha-.35*d):
                continue
            if not (.25*d <= bw <= 2.2*d and .17*d <= bh <= 1.45*d):
                continue
            if not (.05*d*d <= area <= 1.7*d*d):
                continue
            if bh > bw * 2.5:
                continue
            r_x, r_y = max(2,int(.52*d)), max(2,int(.36*d))
            pedazo = tinta[max(0,int(cy)-r_y):min(alto,int(cy)+r_y+1), max(0,int(cx)-r_x):min(ancho,int(cx)+r_x+1)]
            densidad = np.count_nonzero(pedazo)/max(1,pedazo.size)
            if densidad < .2+(100-sensibilidad)*.0012:
                continue
            error = abs((lineas[-1]-cy)/(d/2)-round((lineas[-1]-cy)/(d/2)))
            if error > .40:
                continue
            puntaje = .55*min(densidad/.7,1)+.25*(1-error/.5)+.20*min(area/(.35*d*d),1)
            if puntaje < .39+(100-sensibilidad)*.002:
                continue
            posibles.append((cx,cy,puntaje))
        unicos=[]
        for candidato in sorted(posibles,key=lambda z:z[2],reverse=True):
            if not any(abs(candidato[0]-u[0])<.72*d and abs(candidato[1]-u[1])<.7*d for u in unicos):
                unicos.append(candidato)
        for x,y,p in sorted(unicos):
            nombre,octava=posicion_a_nota(y,s)
            salida.append({'sistema':numero,'x':round(x),'y':round(y),'nota':nombre,
                           'octava':octava,'alteracion_manual':'auto',
                           'origen':'auto','confianza':round(p,2)})
    return salida
