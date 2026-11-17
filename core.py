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



def centro_blanca_cercana(rgb, x_clic, y_clic, pentagrama, radio=1.35):
    """Busca el hueco rodeado por tinta de una nota blanca cerca del clic.

    Una cabeza blanca NO tiene un núcleo negro: las operaciones usadas para las
    negras destruyen su contorno. RETR_TREE localiza el interior claro encerrado
    por el trazo exterior, incluso si la plica está unida a la cabeza.

    Se prioriza un interior ovalado de tamaño musical plausible, se limita la
    búsqueda al clic y al pentagrama, y se comprueba que haya tinta alrededor.
    No analiza el resto de la página ni crea notas automáticamente.
    """
    d = float(pentagrama['distancia'])
    h, w = rgb.shape[:2]
    x_clic, y_clic = float(x_clic), float(y_clic)
    margen = max(12, int(np.ceil(d * 3.5)))
    x0 = max(0, int(x_clic) - margen)
    y0 = max(0, int(y_clic) - margen)
    x1 = min(w, int(x_clic) + margen + 1)
    y1 = min(h, int(y_clic) + margen + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    gris = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    # Un umbral fijo alto mantiene cerrado el óvalo claro aun cuando el clic
    # desplaza el recorte (Otsu local abría y rompía el contorno del mismo óvalo).
    tinta = cv2.threshold(gris, 205, 255, cv2.THRESH_BINARY_INV)[1]
    contornos, jerarquia = cv2.findContours(tinta, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None:
        return None

    propuestas = []
    for i, contorno in enumerate(contornos):
        # Los interiores blancos son hijos de un contorno exterior negro.
        if jerarquia[0][i][3] < 0:
            continue
        bx, by, bw, bh = cv2.boundingRect(contorno)
        area = float(cv2.contourArea(contorno))
        if not (0.65*d <= bw <= 1.8*d and 0.32*d <= bh <= 1.18*d):
            continue
        if not (0.13*d*d <= area <= 0.95*d*d):
            continue
        if not (1.05 <= bw / max(1, bh) <= 3.3):
            continue
        if area / (bw*bh) < 0.32:
            continue

        m = cv2.moments(contorno)
        if m['m00'] <= 0:
            continue
        cx = x0 + m['m10'] / m['m00']
        cy = y0 + m['m01'] / m['m00']
        dx, dy = cx-x_clic, cy-y_clic
        distancia = float(np.hypot(dx, dy))
        if abs(dx) > radio*d or abs(dy) > radio*d or distancia > radio*d:
            continue

        paso = (pentagrama['lineas'][-1] - cy) / (d/2.0)
        error = abs(paso - round(paso))
        # Aceptamos descentrados leves por el contorno grueso y el pentagrama.
        if error > 0.49:
            continue

        # Requerir tinta que rodee al interior: diferencia con los fondos y
        # orificios anómalos en los extremos de una plica.
        # Se prueba un anillo alrededor de la elipse, sin incluir el blanco.
        radx = max(3, int(round(bw / 2 + 1.5)))
        rady = max(3, int(round(bh / 2 + 1.5)))
        xx0, xx1 = max(0, int(cx-x0)-radx), min(tinta.shape[1], int(cx-x0)+radx+1)
        yy0, yy1 = max(0, int(cy-y0)-rady), min(tinta.shape[0], int(cy-y0)+rady+1)
        parche = tinta[yy0:yy1, xx0:xx1]
        tinta_cercana = np.count_nonzero(parche) / max(1, parche.size)
        if tinta_cercana < 0.10:
            continue

        # El interior de una blanca debe seguir siendo mayoritariamente claro.
        # Esto impide que un agujero junto a una negra (o su plica) tenga
        # prioridad sobre la detección normal, que ya funcionaba en V5.1.
        icx, icy = int(round(cx-x0)), int(round(cy-y0))
        centro = gris[max(0,icy-1):min(gris.shape[0],icy+2),
                      max(0,icx-1):min(gris.shape[1],icx+2)]
        centro_amplio = gris[max(0,icy-2):min(gris.shape[0],icy+3),
                             max(0,icx-2):min(gris.shape[1],icx+3)]
        if (np.mean(centro < 170) > 0.70 or
                np.mean(centro_amplio < 170) > 0.70):
            continue

        # Mayor área ovalada es preferible a agujeros secundarios de plicas.
        bonus_forma = 0.25*d * min((area / (bw*bh)) / 0.60, 1.0)
        # Se privilegia el hueco principal, más grande y estable, frente
        # a un segundo agujero artificial creado por la línea del pentagrama.
        puntuacion = 0.36*distancia - 2.8*area/d - 0.20*bonus_forma + 0.10*d*error
        propuestas.append((puntuacion, cx, cy, area))

    if not propuestas:
        return None
    _, cx_principal, cy_principal, _ = min(propuestas)
    # En las notas blancas que atraviesan una línea, el interior se divide
    # en dos huecos. Promediar sus centros evita leer FA como SOL por usar
    # únicamente el hueco superior.
    relacionadas = [p for p in propuestas
                   if abs(p[1]-cx_principal) < 0.60*d
                   and abs(p[2]-cy_principal) < 1.10*d]
    pesos = np.array([p[3] for p in relacionadas], dtype=np.float64)
    cx = float(np.average([p[1] for p in relacionadas], weights=pesos))
    cy = float(np.average([p[2] for p in relacionadas], weights=pesos))
    paso = round((pentagrama['lineas'][-1] - cy) / (d / 2.0))
    cy = pentagrama['lineas'][-1] - paso*d/2.0
    return int(round(cx)), int(round(cy))


def centro_cabeza_cercana(rgb, x_clic, y_clic, pentagrama, radio=1.3):
    """Ajusta el clic al centro de una cabeza, negra o blanca.

    Combina el detector negro de V5.1 con el contorno hueco de una blanca.
    Si ambos proponen un centro, conserva la negra cuando el núcleo es sólido.
    """
    negra = _centro_negra_cercana(rgb, x_clic, y_clic, pentagrama, radio=radio)
    blanca = centro_blanca_cercana(rgb, x_clic, y_clic, pentagrama, radio=radio)
    if negra is not None and blanca is not None:
        # La V5.1 ya detectaba bien las negras. Una negra real presenta
        # un núcleo de 5x5 px prácticamente lleno de tinta; no debemos
        # reemplazarla por un hueco del pentagrama cercano.
        nx, ny = negra
        interior = rgb[max(0,ny-2):min(rgb.shape[0],ny+3),
                       max(0,nx-2):min(rgb.shape[1],nx+3)]
        if interior.size:
            interior = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY)
        if interior.size and float(np.mean(interior < 170)) >= 0.95:
            return negra
        return blanca
    return blanca if blanca is not None else negra


def _centro_negra_cercana(rgb, x_clic, y_clic, pentagrama, radio=1.3):
    """Ajusta un clic cercano al centro de una cabeza de nota real.

    Busca componentes ovalados gruesos en una zona pequeña de la imagen, en lugar
    de calcular la altura musical usando directamente la posición vertical del clic.
    Si la imagen no aporta un candidato cercano y convincente, devuelve None.

    El ajuste asistido es conservador y NO sustituye a un reconocedor OMR completo.
    """
    d = float(pentagrama['distancia'])
    alto, ancho = rgb.shape[:2]
    x_clic, y_clic = float(x_clic), float(y_clic)
    r = max(8, int(np.ceil(d * 3.4)))
    x0 = max(0, int(x_clic) - r)
    y0 = max(0, int(y_clic) - r)
    x1 = min(ancho, int(x_clic) + r + 1)
    y1 = min(alto, int(y_clic) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    gris = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    _, tinta = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Lascas horizontales del pentagrama y plicas finas desaparecen al abrir.
    # Un óvalo grueso suele sobrevivir.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gruesa = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, kernel)
    total, _, estadisticas, centros = cv2.connectedComponentsWithStats(gruesa, 8)
    opciones = []
    for i in range(1, total):
        bx, by, bw, bh, area = [int(v) for v in estadisticas[i]]
        cx, cy = float(centros[i][0] + x0), float(centros[i][1] + y0)
        # Un óvalo puede tocar una plica, pero no debe parecer un segmento largo.
        if not (0.55*d <= bw <= 2.25*d and 0.36*d <= bh <= 1.70*d):
            continue
        if not (0.28*d*d <= area <= 2.7*d*d):
            continue
        if not (0.55 <= bw/max(1,bh) <= 2.9):
            continue
        # Verificar que el centro coincide aproximadamente con una línea/espacio.
        paso = (pentagrama['lineas'][-1] - cy)/(d/2)
        err = abs(paso-round(paso))
        if err > 0.43:
            continue
        dx, dy = cx - x_clic, cy - y_clic
        distancia = np.hypot(dx, dy)
        # Permitir un poco de error vertical, sin saltar a notas distantes.
        if abs(dx) > radio*d or abs(dy) > radio*d or distancia > radio*d:
            continue
        # No elegir un fragmento más lejano solo por ser grande.
        opciones.append((distancia + err*0.08*d, cx, cy))

    if not opciones:
        return None
    _, cx, cy = min(opciones)
    # Ajustamos la posición vertical al grado musical detectado; el eje X procede
    # de la cabeza de la nota. De ese modo un clic ligeramente alto da igual.
    paso = round((pentagrama['lineas'][-1] - cy)/(d/2))
    y_afinada = pentagrama['lineas'][-1] - paso*(d/2)
    return int(round(cx)), int(round(y_afinada))
