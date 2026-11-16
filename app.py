import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

NOTES=['DO','RE','MI','FA','SOL','LA','SI']
STEPS=['MI','FA','SOL','LA','SI','DO','RE']
COLS=['sistema','compas_estimado','x','y','nota','octava','alteracion','confianza']

def clusters(values,gap=2):
    groups=[]
    for value in values:
        value=int(value)
        if not groups or value-groups[-1][-1]>gap: groups.append([value])
        else: groups[-1].append(value)
    return [int(np.median(g)) for g in groups]

def find_staffs(gray):
    h,w=gray.shape
    binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(25,w//11),1)))
    coverage=np.count_nonzero(horizontal,axis=1)
    ys=clusters(np.flatnonzero(coverage>w*.28),max(2,h//1000))
    staffs=[]
    for i in range(len(ys)-4):
        five=ys[i:i+5]
        ds=np.diff(five)
        d=float(np.median(ds))
        if 6<=d<=65 and np.max(np.abs(ds-d))<=max(2.5,d*.22):
            if not staffs or five[0]>staffs[-1][-1]+2*d: staffs.append(five)
    return binary,horizontal,staffs

def pitch(y,staff):
    d=float(np.median(np.diff(staff)))
    step=int(round((staff[-1]-y)/(d/2)))
    # Bottom line is E4, every half-spacing is one diatonic step.
    idx=step%7
    octave=4+(step+2)//7
    return STEPS[idx],octave

def extent(horizontal,staff):
    h,w=horizontal.shape
    mask=np.any(horizontal[max(0,staff[0]-2):min(h,staff[-1]+3)]>0,axis=0)
    runs=[]
    for x in np.flatnonzero(mask):
        if not runs or x-runs[-1][-1]>5:runs.append([int(x)])
        else:runs[-1].append(int(x))
    return (min(max(runs,key=len)),max(max(runs,key=len))) if runs else (0,w-1)

def detect(gray,binary,horizontal,staffs,sensitivity):
    h,w=gray.shape
    # Retain V1-style broad contour candidates. Suppress staff lines only for candidate extraction.
    linefree=cv2.subtract(binary,horizontal)
    output=[]
    for sn,staff in enumerate(staffs,1):
        d=float(np.median(np.diff(staff)))
        left,right=extent(horizontal,staff)
        # Skip clef, key signature and time signature at each system start.
        start=left+(6.2 if sn==1 else 3.2)*d
        top=max(0,int(staff[0]-3.3*d));bottom=min(h,int(staff[-1]+3.3*d))
        roi=linefree[top:bottom]
        n,labels,stats,centers=cv2.connectedComponentsWithStats(roi)
        cand=[]
        for k in range(1,n):
            x,y,bw,bh,area=stats[k]
            cx,cy=float(centers[k][0]),float(centers[k][1]+top)
            if not(start<cx<right-d*.35):continue
            # Broad bounds restore notes rejected by v2, including fragments on staff lines.
            if not (.22*d<=bw<=2.2*d and .14*d<=bh<=1.55*d):continue
            if not (.035*d*d<=area<=1.8*d*d):continue
            if bh>bw*2.6:continue
            # Check original image around predicted note center for oval black core.
            rx=max(2,int(.53*d));ry=max(2,int(.37*d))
            patch=binary[max(0,int(cy)-ry):min(h,int(cy)+ry+1),max(0,int(cx)-rx):min(w,int(cx)+rx+1)]
            density=np.count_nonzero(patch)/max(1,patch.size)
            if density<(.18+(100-sensitivity)*.0012):continue
            grid_err=abs((staff[-1]-cy)/(d/2)-round((staff[-1]-cy)/(d/2)))
            if grid_err>.49:continue
            score=.55*min(density/.7,1)+.25*(1-grid_err/.5)+.20*min(area/(.35*d*d),1)
            if score < .33+(100-sensitivity)*.002:continue
            cand.append((cx,cy,score))
        # Deduplicate, preserving strongest candidates.
        unique=[]
        for c in sorted(cand,key=lambda a:a[2],reverse=True):
            if not any(abs(c[0]-u[0])<.65*d and abs(c[1]-u[1])<.60*d for u in unique):unique.append(c)
        # Detect narrow full-height barlines (approximate compasses).
        region=binary[max(0,staff[0]-1):min(h,staff[-1]+2)]
        bars=clusters(np.flatnonzero(np.count_nonzero(region,axis=0)>.91*region.shape[0]),max(1,int(d*.15)))
        bars=[b for b in bars if start<b<right-d]
        for cx,cy,score in sorted(unique,key=lambda c:c[0]):
            if any(abs(cx-b)<.4*d for b in bars):continue
            name,octave=pitch(cy,staff)
            bar=1+sum(b<cx for b in bars)
            output.append([sn,bar,round(cx),round(cy),name,octave,'',round(score,2)])
    return pd.DataFrame(output,columns=COLS)

def annotate(rgb,df):
    im=Image.fromarray(rgb).convert('RGB');draw=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',max(12,min(22,im.width//90)))
    except OSError:font=ImageFont.load_default()
    for _,r in df.iterrows():
        try:
            x,y=int(r.x),int(r.y)
            if not(0<=x<im.width and 0<=y<im.height):continue
            label=str(r.nota)+str(r.alteracion if pd.notna(r.alteracion) else '')
            bb=draw.textbbox((0,0),label,font=font);tw=bb[2]-bb[0];th=bb[3]-bb[1]
            tx=max(0,x-tw//2);ty=max(0,y-th-22)
            draw.rectangle((tx-2,ty-2,tx+tw+2,ty+th+2),fill='white')
            draw.text((tx,ty),label,font=font,fill='#1354aa')
            draw.ellipse((x-3,y-3,x+3,y+3),outline='#e33b30',width=2)
        except (TypeError,ValueError,AttributeError):pass
    return im

st.set_page_config(page_title='Lector de partituras V3',layout='wide')
st.title('🎼 Lector de partituras — DO RE MI · V3')
st.caption('Detector geométrico experimental. Recupera detecciones amplias; las notas y compases deben revisarse.')
sensitivity=st.slider('Sensibilidad',20,100,65,5)
file=st.file_uploader('Sube JPG o PNG',type=['jpg','jpeg','png'])
if file:
    rgb=np.asarray(Image.open(file).convert('RGB'))
    if max(rgb.shape[:2])>2600:
        scale=2600/max(rgb.shape[:2]);rgb=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    binary,horizontal,staffs=find_staffs(gray)
    st.write('Pentagramas detectados:',len(staffs))
    if not staffs:st.error('No se detectaron pentagramas. Prueba un recorte recto y nítido.');st.stop()
    df=detect(gray,binary,horizontal,staffs,sensitivity)
    st.write('Candidatos detectados:',len(df))
    threshold=st.slider('Mostrar detecciones desde confianza',0.,1.,0.,.05)
    if threshold>0:df=df[df.confianza>=threshold].copy()
    st.subheader('1. Revisa y corrige las notas')
    st.info('No se interpretan automáticamente armaduras ni alteraciones; compases aproximados. Puedes agregar o borrar filas.')
    edited=st.data_editor(df,num_rows='dynamic',hide_index=True,use_container_width=True,column_config={'nota':st.column_config.SelectboxColumn('Nota',options=NOTES),'alteracion':st.column_config.SelectboxColumn('Alteración',options=['','♯','♭','♮'])},key=f'edit_{file.name}_{sensitivity}_{threshold}')
    st.subheader('2. Imagen anotada')
    result=annotate(rgb,edited);st.image(result,use_container_width=True)
    buff=io.BytesIO();result.save(buff,format='PNG')
    st.download_button('Descargar PNG',buff.getvalue(),'partitura_v3.png','image/png')
    st.subheader('3. Notas por compás')
    if len(edited):
        safe=edited.dropna(subset=['sistema','compas_estimado','x','nota']).copy()
        safe['texto']=safe.nota.astype(str)+safe.alteracion.fillna('').astype(str)
        grouped=(safe.sort_values(['sistema','compas_estimado','x']).groupby(['sistema','compas_estimado'],sort=True)['texto'].apply(lambda v:' – '.join(v)).reset_index())
        st.dataframe(grouped,use_container_width=True,hide_index=True)
        st.download_button('Descargar CSV',grouped.to_csv(index=False).encode('utf-8-sig'),'notas_v3.csv','text/csv')
