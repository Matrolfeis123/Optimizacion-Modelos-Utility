import pulp
import pandas as pd
import pandas as pd
import datetime
from funciones_visualizacion_resultados import menu_graficos
from funciones_extra import calcular_ventanas_carga_descarga_año, calcular_ventanas_carga_descarga_diario, calcular_ventanas_carga_descarga_diario_v2, calcular_ventanas_carga_descarga_diario_v3

## Este modelo, a diferencia del anterior, considera nuevas restricciones y variables:



# Vamos a implementar la optimización de carga y descarga de baterías en mi planta solar, primeramente para el mes de enero
# Cargar el archivo CSV de costos marginales
costos_marginales_df = pd.read_csv('CMg.csv', sep=';', decimal=',')

costos_marginales_df['Año'] = costos_marginales_df['Año'].astype(int)
costos_marginales_df['Mes'] = costos_marginales_df['Mes'].astype(int)
costos_marginales_df['Hora'] = costos_marginales_df['Hora'].astype(int)
costos_marginales_df['CMg'] = costos_marginales_df['CMg'].astype(float)


# Definir las fechas de inicio y fin
fecha_inicio = datetime.datetime(costos_marginales_df['Año'].min(), 1, 1, 0)  # Año inicial, 1 de enero, 00:00 horas
fecha_fin = datetime.datetime(costos_marginales_df['Año'].max(), 12, 31, 23)  # Año final, 31 de diciembre, 23:00 horas
rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='H')
df_fechas = pd.DataFrame({'FechaHora': rango_fechas})
# Extraer Año, Mes, Día y Hora
df_fechas['Año'] = df_fechas['FechaHora'].dt.year
df_fechas['Mes'] = df_fechas['FechaHora'].dt.month
df_fechas['Día'] = df_fechas['FechaHora'].dt.day
df_fechas['Hora'] = df_fechas['FechaHora'].dt.hour

#Eliminar 29 de febrero para todo año que sea bisiesto
df_fechas = df_fechas[~((df_fechas['Mes'] == 2) & (df_fechas['Día'] == 29))]

# Unir los datos de costos marginales al DataFrame de fechas
df_cmg = pd.merge(df_fechas, costos_marginales_df, on=['Año', 'Mes', 'Hora'], how='left')

# Verificar si hay valores nulos en CMg
print(df_cmg['CMg'].isnull().sum())

# Optimizar tipos de datos
df_cmg['Año'] = df_cmg['Año'].astype('int16')
df_cmg['Mes'] = df_cmg['Mes'].astype('int8')
df_cmg['Día'] = df_cmg['Día'].astype('int8')
df_cmg['Hora'] = df_cmg['Hora'].astype('int8')
df_cmg['CMg'] = df_cmg['CMg'].astype('float32')

# Eliminar la columna 'FechaHora' si no es necesaria
df_cmg.drop('FechaHora', axis=1, inplace=True)




# Parámetros de la planta

# PV inicial
peak_power = 10.8  # MW
nominal_power = 9  # MW
CoD = 2028 # 
project_lifetime = 25
pv_degradation = 0
inverter_efficency_pv = 1
# BESS
bess_charge_power = 9  # MW
bess_discharge_power = 9  # MW
bess_charge_hours = 6
bess_discharge_hours = 4
bess_initial_energy_capacity = 36  # MWh
bess_degradation = 0
bess_charge_efficency = 1
bess_discharge_efficency = 1
inverter_efficency_bess = 1
carga_min_bess = 0

# Restricciones de operación
# hora_inicio_carga = 9
# hora_fin_carga = 18
# hora_inicio_descarga = 19
# hora_fin_descarga = 8


# vamos a elegir el año correspondiente al CoD
df_cmg_año1 = df_cmg[df_cmg['Año'] == CoD]
costos_marginales = df_cmg_año1['CMg'].tolist() # es la lista
costos_mg_año1 = costos_marginales

# Extraer la lista de costos marginales
# costos_marginales = df_cmg['CMg'].tolist()

# costos_mg_año1 = costos_marginales[0:8760]

# Vamos a importar el vector de generación de energía solar desde el CSV 'generacion.csv'
generacion = pd.read_csv('generacion.csv', sep=';')
generacion = generacion['G solar'].tolist()
for i in range(len(generacion)):
    generacion[i] = generacion[i].replace(',', '.')

for i in range(len(generacion)):
    generacion[i] = max(0.0, float(generacion[i]))

# Comprobamos las dimensiones de los vectores de costo marginal y generación
print("dim CMg: ", len(costos_marginales))
print("Dim vector generacion: ", len(generacion))


# Definir el número total de horas: Aca, estamos utilizando la totalidad de los datos, probablemente sea mejor ir llamando a los datos año a año,para optimizar el uso de memoria
T = len(costos_marginales)
t_indices = range(T)


# Definimos el modelo de optimización
model = pulp.LpProblem('Solar_PV_BESS_Optimization', pulp.LpMaximize)

# Variables de decisión
C_t = pulp.LpVariable.dicts('C_t', t_indices, lowBound=0, upBound=bess_charge_power, cat=pulp.LpContinuous)
D_t = pulp.LpVariable.dicts('D_t', t_indices, lowBound=0, upBound=bess_discharge_power, cat=pulp.LpContinuous)
SOC_t = pulp.LpVariable.dicts('SOC_t', t_indices, lowBound=0, upBound=bess_initial_energy_capacity, cat=pulp.LpContinuous)

# Variables binarias para estado de carga y descarga
charge_status = pulp.LpVariable.dicts("ChargeStatus", t_indices, cat='Binary')
discharge_status = pulp.LpVariable.dicts("DischargeStatus", t_indices, cat='Binary')

# Variables adicionales para inyección a la red y curtailment
PV_grid_t = pulp.LpVariable.dicts('PV_grid_t', t_indices, lowBound=0, upBound=nominal_power, cat=pulp.LpContinuous)
PV_curtail_t = pulp.LpVariable.dicts('PV_curtail_t', t_indices, lowBound=0, cat=pulp.LpContinuous)

# Actualizamos la función objetivo para incluir penalizaciones
revenue_terms = []
for t in t_indices:
    # Revenue from PV generation sold to the grid
    revenue_pv = PV_grid_t[t] * costos_mg_año1[t]
    # Revenue from BESS discharge
    revenue_bess = D_t[t] * costos_mg_año1[t]
    # Cost of charging the BESS (opportunity cost)
    cost_charge = C_t[t] * costos_mg_año1[t] * 0.1
    # Penalization for curtailment
    penalty_curtail = PV_curtail_t[t] * 0.01
    # Total revenue at time t
    revenue_terms.append(revenue_pv + 5*revenue_bess - cost_charge - penalty_curtail)

# Establecer la función objetivo
model += pulp.lpSum(revenue_terms)
# RESTRICCIONES:

G_pv_t = [
    generacion[t]*inverter_efficency_pv*(1-pv_degradation)**(t//8760)
    for t in t_indices
]


# 1. Balance de generación solar
for t in t_indices:
    model += (
        G_pv_t[t] == PV_grid_t[t] + C_t[t] + PV_curtail_t[t],
        f"PV_Generation_Balance_{t}"
    )

# 2. Limitar la inyección neta al grid a la potencia nominal
for t in t_indices:
    model += (
        PV_grid_t[t] + D_t[t] <= nominal_power,
        f"Max_Net_Injection_{t}"
    )

for t in t_indices:
    model += (
        PV_grid_t[t] + C_t[t] <= peak_power,
        f"Max_PV_Utilization_{t}"
    )

# 3. Limites de carga y descarga
for t in t_indices:
    model += (
        C_t[t] <= bess_charge_power,
        f"Max_Charge_{t}"
    )
    model += (
        D_t[t] <= bess_discharge_power,
        f"Max_Discharge_{t}"
    )

# 4. Flujo del Estado de Carga (SoC)
model += (
    SOC_t[0] == 0 + C_t[0] * bess_charge_efficency * inverter_efficency_bess - D_t[0] * (1 / (bess_discharge_efficency * inverter_efficency_bess)),
    'SOC_initial_condition'
)

for t in t_indices:
    if t > 0:
        model += (
            SOC_t[t] == SOC_t[t - 1] + 
            C_t[t] * bess_charge_efficency * inverter_efficency_bess - 
            D_t[t] * (1 / (bess_discharge_efficency * inverter_efficency_bess)),
            f"StateOfCharge_{t}"
        )

# 5. Limites del SoC
for t in t_indices:
    model += (
        SOC_t[t] <= bess_initial_energy_capacity * (1 - bess_degradation)**(t//8760),
        f"Max_SOC_{t}"
    )
    model += (
        SOC_t[t] >= carga_min_bess,
        f"Min_SOC_{t}"
    )

# 6. No simultaneidad de carga y descarga
big_M = max(bess_charge_power, bess_discharge_power)
for t in t_indices:
    model += (
        C_t[t] <= big_M * charge_status[t],
        f"Charge_Big_M_{t}"
    )
    model += (
        D_t[t] <= big_M * discharge_status[t],
        f"Discharge_Big_M_{t}"
    )
    model += (
        charge_status[t] + discharge_status[t] <= 1,
        f"Charge_Discharge_Exclusivity_{t}"
    )

# 7. Ventanas Horarias de Operación
# Asumiendo que tienes la función calcular_ventanas_carga_descarga_diario actualizada
horas_carga_por_dia = int(bess_charge_hours)
horas_descarga_por_dia = int(bess_discharge_hours)

# Calcular las ventanas horarias
ventanas_carga, ventanas_descarga = calcular_ventanas_carga_descarga_diario_v3(df_cmg_año1, generacion)

# Añadir restricciones al modelo
for t in t_indices:
    mes = df_cmg_año1.iloc[t]['Mes']
    dia = df_cmg_año1.iloc[t]['Día']
    hora = df_cmg_año1.iloc[t]['Hora']
    
    # Obtener las horas de carga y descarga para el día actual
    horas_carga_dia = ventanas_carga.get((mes, dia), [])
    horas_descarga_dia = ventanas_descarga.get((mes, dia), [])
    
    # Restricciones de carga
    if hora in horas_carga_dia:
        pass
    else:
        model += (C_t[t] == 0, f"Charge_Hours_{t}")
        model += (charge_status[t] == 0, f"Charge_Status_Hours_{t}")
    
    # Restricciones de descarga
    if hora in horas_descarga_dia:
        pass
    else:
        model += (D_t[t] == 0, f"Discharge_Hours_{t}")
        model += (discharge_status[t] == 0, f"Discharge_Status_Hours_{t}")

# Resolver el modelo
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print(pulp.LpStatus[model.status])

# Si la solución es óptima, puedes extraer y analizar los resultados
if pulp.LpStatus[model.status] == 'Optimal':
    C_sol = [C_t[t].varValue for t in t_indices]
    D_sol = [D_t[t].varValue for t in t_indices]
    SOC_sol = [SOC_t[t].varValue for t in t_indices]
    PV_grid_sol = [PV_grid_t[t].varValue for t in t_indices]
    PV_curtail_sol = [PV_curtail_t[t].varValue for t in t_indices]

    results = pd.DataFrame({
        'Hora': [t for t in t_indices],
        'CMg': [costos_mg_año1[t] for t in t_indices],
        'Generacion_PV_sin_Degr': [generacion[t] for t in t_indices],
        'Generacion_PV': [G_pv_t[t] for t in t_indices],
        'PV_Inyectada_Grid': PV_grid_sol,
        'PV_Curtailment': PV_curtail_sol,
        'Carga_BESS': C_sol,
        'Descarga_BESS': D_sol,
        'SOC': SOC_sol,
    })

    # Añadir columnas de 'Año', 'Mes', 'Día' y 'Hora_del_día' desde 'df_cmg_año1'
    results['Año'] = df_cmg_año1['Año'].values
    results['Mes'] = df_cmg_año1['Mes'].values
    results['Día'] = df_cmg_año1['Día'].values
    results['Hora_del_día'] = df_cmg_año1['Hora'].values

    # Calcular el despacho neto al grid
    results['Despacho_Neto_Grid'] = results['PV_Inyectada_Grid'] + results['Descarga_BESS']

    # Mostrar los primeros registros
    print(results.head())
    results.to_excel('resultados_optimización.xlsx', index=False)
    
    # Aquí puedes llamar a tus funciones de gráficos
    menu_graficos(results)
else:
    print('La optimización no fue exitosa. Estado:', pulp.LpStatus[model.status])
#Para integrar los resultados al excel, copiar y pegar los vectores correspondientes.
# si eso funciona, haremos una hoja extra donde vaya a buscar estos valores y haga los calculos.
# Tambien, deberia comparar los resultados del VAN esta optimizacion vs la que se usaba antes, para ver el impacto