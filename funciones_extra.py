from itertools import groupby
from operator import itemgetter
    

def calcular_ventanas_carga_descarga_diario_v3(df_cmg_año1, generacion):
    # Si bien esta forma funciona, no es la más eficiente
    # Podria ser mas eficiente solo corroborar que:
    # 1. para carga, la generacion sea mayor a 0 y que el CMg el minimo
    # 2. para descarga, el CMg sea el maximo
    # 3. Dar vuelta

    # Añadir la generación al DataFrame
    df_cmg_año1 = df_cmg_año1.copy()
    df_cmg_año1['Generacion'] = generacion
    print(df_cmg_año1.head(24))
    # Crear diccionarios de ventanas
    ventanas_carga = {}
    ventanas_descarga = {}
    for mes in df_cmg_año1['Mes'].unique():
        for dia in df_cmg_año1['Día'].unique():
            df_dia = df_cmg_año1[(df_cmg_año1['Mes'] == mes) & (df_cmg_año1['Día'] == dia)]
            df_dia = df_dia.sort_values('Hora')
            
            # Cargar horas de carga
            horas_carga = df_dia[df_dia['Generacion'] > 0.2]['Hora'].tolist()
            horas_carga = [hora for hora in horas_carga]
            
            # Cargar horas de descarga
            horas_descarga = df_dia[df_dia['Generacion'] < 0.2]['Hora'].tolist()
            horas_descarga = [hora for hora in horas_descarga if hora not in horas_carga]
            
            # Guardar en los diccionarios
            ventanas_carga[(mes, dia)] = horas_carga
            ventanas_descarga[(mes, dia)] = horas_descarga

    return ventanas_carga, ventanas_descarga


def calcular_ventanas_carga_descarga_diario_v2(df_cmg_año1, generacion):
    # Si bien esta forma funciona, no es la más eficiente
    # Podria ser mas eficiente solo corroborar que:
    # 1. para carga, la generacion sea mayor a 0 y que el CMg el minimo
    # 2. para descarga, el CMg sea el maximo
    # 3. Dar vuelta carga score y dividir CMg en generacion, para decidir en $/MWh

    # Añadir la generación al DataFrame
    df_cmg_año1 = df_cmg_año1.copy()
    df_cmg_año1['Generacion'] = generacion
    print(df_cmg_año1.head(24))
    
    # Cálculo de una métrica combinada
    df_cmg_año1['Carga_Score'] = df_cmg_año1['Generacion'] / df_cmg_año1['CMg']
    df_cmg_año1['Descarga_Score'] = df_cmg_año1['CMg']
    
    # Si hay valores inf, reemplazarlos por df_cmg_año1['Generacion']
    df_cmg_año1['Carga_Score'] = df_cmg_año1['Carga_Score'].replace([float('inf')], 0.5)
    
    # Normalizar los scores
    df_cmg_año1['Carga_Score'] = df_cmg_año1['Carga_Score'] / df_cmg_año1['Carga_Score'].max()
    
    # imprimir los primeros 24 cargar score con 3 decimales
    print(df_cmg_año1['Carga_Score'].head(24).round(3))

    df_cmg_año1['Descarga_Score'] = df_cmg_año1['Descarga_Score'] / df_cmg_año1['Descarga_Score'].max()

    print(df_cmg_año1['Descarga_Score'].head(24).round(3))
    
    # Seleccionar las horas top para carga y descarga
    umbral_carga = 0  # Puedes ajustar este valor
    umbral_descarga = 0.5 # Puedes ajustar este valor
    
    df_cmg_año1['Es_Carga'] = df_cmg_año1['Carga_Score'] >= umbral_carga
    df_cmg_año1['Es_Descarga'] = df_cmg_año1['Descarga_Score'] >= umbral_descarga
    
    # Evitar superposición
    df_cmg_año1.loc[df_cmg_año1['Es_Carga'] & df_cmg_año1['Es_Descarga'], 'Es_Carga'] = False

    #Imprimir Carga_Score y Descarga_Score promedio mensual
    print(df_cmg_año1.groupby('Mes')['Carga_Score'].mean().round(3))
    print(df_cmg_año1.groupby('Mes')['Descarga_Score'].mean().round(3))
    
    # Crear diccionarios de ventanas
    ventanas_carga = {}
    ventanas_descarga = {}
    
    for idx, row in df_cmg_año1.iterrows():
        mes = row['Mes']
        dia = row['Día']
        hora = row['Hora']
        
        key = (mes, dia)
        
        if row['Es_Carga']:
            ventanas_carga.setdefault(key, []).append(hora)
        if row['Es_Descarga']:
            ventanas_descarga.setdefault(key, []).append(hora)
    
    return ventanas_carga, ventanas_descarga

def calcular_ventanas_carga_descarga_diario(df_cmg_año1, horas_carga_por_dia, horas_descarga_por_dia, generacion):

    # Diccionarios para almacenar las ventanas de carga y descarga por día
    ventanas_carga = {}
    ventanas_descarga = {}
    
    # Añadir la generación al DataFrame
    df_cmg_año1 = df_cmg_año1.copy()
    df_cmg_año1['Generacion'] = generacion
    
    # Asegurarnos de que el DataFrame esté ordenado por fecha y hora
    df_cmg_año1 = df_cmg_año1.sort_values(['Mes', 'Día', 'Hora']).reset_index(drop=True)
    
    # Agrupar por día
    dias_unicos = df_cmg_año1[['Mes', 'Día']].drop_duplicates()
    
    for idx, row in dias_unicos.iterrows():
        mes = row['Mes']
        dia = row['Día']
        
        # Filtrar datos para el día actual
        df_dia = df_cmg_año1[(df_cmg_año1['Mes'] == mes) & (df_cmg_año1['Día'] == dia)]
        
        # Asegurarnos de que hay suficientes datos
        if len(df_dia) == 0:
            continue
        
        # Horas disponibles con generación para carga
        df_carga = df_dia[df_dia['Generacion'] > 0]
        df_carga = df_carga.sort_values('Hora')
        
        # Encontrar bloques consecutivos de horas de carga
        horas_carga_disponibles = df_carga['Hora'].tolist()
        bloques_carga = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(horas_carga_disponibles), lambda x: x[0]-x[1])]
        
        # Seleccionar el bloque con el menor promedio de CMg
        mejor_bloque_carga = None
        menor_cmg_promedio = float('inf')
        for bloque in bloques_carga:
            if len(bloque) >= horas_carga_por_dia:
                for i in range(len(bloque) - horas_carga_por_dia + 1):
                    sub_bloque = bloque[i:i+horas_carga_por_dia]
                    cmg_promedio = df_carga[df_carga['Hora'].isin(sub_bloque)]['CMg'].mean()
                    if cmg_promedio < menor_cmg_promedio:
                        menor_cmg_promedio = cmg_promedio
                        mejor_bloque_carga = sub_bloque
        # Si no se encontró un bloque suficientemente largo, tomar el más largo disponible
        if not mejor_bloque_carga:
            mejor_bloque_carga = max(bloques_carga, key=len)[:horas_carga_por_dia]
        
        # Horas para descarga (sin considerar generación)
        df_descarga = df_dia.sort_values('Hora')
        horas_descarga_disponibles = df_descarga['Hora'].tolist()
        bloques_descarga = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(horas_descarga_disponibles), lambda x: x[0]-x[1])]
        
        # Encontrar bloques consecutivos de descarga con los mayores CMg
        mejor_bloque_descarga = None
        mayor_cmg_promedio = float('-inf')
        for bloque in bloques_descarga:
            if len(bloque) >= horas_descarga_por_dia:
                for i in range(len(bloque) - horas_descarga_por_dia + 1):
                    sub_bloque = bloque[i:i+horas_descarga_por_dia]
                    cmg_promedio = df_descarga[df_descarga['Hora'].isin(sub_bloque)]['CMg'].mean()
                    if cmg_promedio > mayor_cmg_promedio:
                        mayor_cmg_promedio = cmg_promedio
                        mejor_bloque_descarga = sub_bloque
        # Si no se encontró un bloque suficientemente largo, tomar el más largo disponible
        if not mejor_bloque_descarga:
            mejor_bloque_descarga = max(bloques_descarga, key=len)[:horas_descarga_por_dia]
        
        # Asegurarnos de que no haya superposición
        mejor_bloque_carga = [hora for hora in mejor_bloque_carga if hora not in mejor_bloque_descarga]
        mejor_bloque_descarga = [hora for hora in mejor_bloque_descarga if hora not in mejor_bloque_carga]
        
        # Guardar en los diccionarios
        ventanas_carga[(mes, dia)] = mejor_bloque_carga
        ventanas_descarga[(mes, dia)] = mejor_bloque_descarga
    
    return ventanas_carga, ventanas_descarga

def calcular_ventanas_carga_descarga_año(df_cmg):
    """
    Calcula las ventanas horarias de carga para cada mes y define las ventanas de descarga
    como el complemento de las horas de carga.

    Parámetros:
    df_cmg: DataFrame que contiene las columnas 'Mes', 'Hora' y 'CMg'.

    Retorna:
    - charging_intervals: Diccionario con las horas de carga por mes.
    - discharging_intervals: Diccionario con las horas de descarga por mes (complemento de carga).
    """
    charging_intervals = {}
    discharging_intervals = {}
    
    # Obtener la lista de meses únicos
    meses_unicos = df_cmg['Mes'].unique()
    
    for mes in meses_unicos:
        # Filtrar los datos para el mes actual
        df_mes = df_cmg[df_cmg['Mes'] == mes]
        
        # Calcular el umbral bajo (por ejemplo, percentil 25) para identificar las horas de carga
        umbral_bajo = df_mes['CMg'].quantile(0.25)
        
        # Determinar las horas de carga (costos marginales menores o iguales al umbral bajo)
        horas_carga = df_mes[df_mes['CMg'] <= umbral_bajo]['Hora'].unique().tolist()
        
        # Las horas de descarga son todas las horas que no están en las horas de carga
        todas_las_horas = list(range(24))
        horas_descarga = [hora for hora in todas_las_horas if hora not in horas_carga]
        
        # Ordenar las listas de horas
        horas_carga.sort()
        horas_descarga.sort()
        
        charging_intervals[mes] = horas_carga
        discharging_intervals[mes] = horas_descarga
        
    return charging_intervals, discharging_intervals
