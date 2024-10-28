import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Vamos a implementar la optimización de carga y descarga de baterías en mi planta solar, primeramente para el mes de enero


costos_marginales_diarios = [
    108.2, 103.4, 100.4, 100.4, 100.4, 100.4,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 123.4, 127.9,
    125.7, 125.7
]

# Vamos a extender el vector de costos marginales diarios para todo el mes de enero
horas_enero = 24 * 31
costos_marginales_enero = costos_marginales_diarios * 31

# print(costos_marginales_enero)

# Vamos a importar el vector de generación de energía solar desde el CSV 'generacion.csv'
generacion = pd.read_csv('generacion.csv', sep=';')

# Procesamiento de los datos de generación
# vamos a guardar los datos de generacion en una lista, unicamente la columna de generacion como integer
generacion = generacion['G solar'].tolist()
for i in range(len(generacion)):
    generacion[i] = generacion[i].replace(',', '.')

for i in range(len(generacion)):
    generacion[i] = max(0.0, float(generacion[i]))

# Parámetros de la planta

# PV
peak_power = 10.8  # MW
nominal_power = 9  # MW
CoD = 2028 # 
project_lifetime = 25
pv_degradation = 0

# BESS
bess_charge_power = 9  # MW
bess_discharge_power = 9  # MW
bess_charge_hours = 4
bess_discharge_hours = 4
bess_energy_capacity = bess_charge_power * bess_charge_hours  # MWh
bess_degradation = 0
bess_charge_efficency = 1
bess_discharge_efficency = 1
inverter_efficency = 1

# Restricciones de operación
hora_inicio_carga = 9
hora_fin_carga = 18
hora_inicio_descarga = 19
hora_fin_descarga = 8

model = pulp.LpProblem('Solar_PV_BESS_Optimization', pulp.LpMaximize)

# Definición de conjuntos e índices
T = horas_enero
t_indices = range(T)

años_desde_cod = 0  # Para enero

# Variables de decisión
C_t = pulp.LpVariable.dicts('C_t', t_indices, lowBound=0, upBound=bess_charge_power, cat=pulp.LpContinuous)
D_t = pulp.LpVariable.dicts('D_t', t_indices, lowBound=0, upBound=bess_discharge_power, cat=pulp.LpContinuous)
SOC_t = pulp.LpVariable.dicts('SOC_t', t_indices, lowBound=0, upBound=bess_energy_capacity, cat=pulp.LpContinuous)
# Variables binarias para estado de carga y descarga
charge_status = pulp.LpVariable.dicts("ChargeStatus", t_indices, cat='Binary')
discharge_status = pulp.LpVariable.dicts("DischargeStatus", t_indices, cat='Binary')

generacion_pv = [
    generacion[t] * inverter_efficency * (1 - pv_degradation) ** años_desde_cod
    for t in t_indices
]

# Función objetivo: maximizar beneficios
revenue_terms = []
for t in t_indices:
    # Revenue from PV generation sold to the grid
    revenue_pv = (generacion_pv[t] - C_t[t]) * costos_marginales_enero[t]
    # Revenue from BESS discharge
    revenue_bess = D_t[t] * costos_marginales_enero[t]
    # Total revenue at time t
    revenue_terms.append(revenue_pv + revenue_bess)



# Set the objective function
model += pulp.lpSum(revenue_terms)
# Restricciones

# 1) Dinámica del estado de carga (SOC)
model += (
    SOC_t[0] == 0 + C_t[0] * bess_charge_efficency * inverter_efficency - D_t[0] * (1 / (bess_discharge_efficency * inverter_efficency)),
    'SOC_initial_condition'
)

for t in t_indices:
    if t > 0:
        model += (
            SOC_t[t] == SOC_t[t - 1] + 
            C_t[t] * bess_charge_efficency * inverter_efficency - 
            D_t[t] * (1 / (bess_discharge_efficency * inverter_efficency)),
            f"StateOfCharge_{t}"
        )
# 2) Límites del estado de carga (SOC)

for t in t_indices:
    model += (
        SOC_t[t] <= bess_energy_capacity,
        f"Max_SOC_{t}"
    )

    model += (
        SOC_t[t] >= 0,
        f"Min_SOC_{t}"
    )

# 3) Límites de carga y descarga
for t in t_indices:
    model += (
        C_t[t] <= bess_charge_power,
        f"Max_Charge_{t}"
    )

    model += (
        D_t[t] <= bess_discharge_power,
        f"Max_Discharge_{t}"
    )

# 4) Restricción de carga máxima (evitar cargar más de la generación disponible)
for t in t_indices:
    model += (
        C_t[t] <= generacion_pv[t],
        f"Charge_Limit_{t}"
    )



# Big M para las restricciones
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

# 6) Restricciones de horas de operación
for t in t_indices:
    hora = t % 24

    # Horas de carga: 7 AM - 18 PM
    if hora_inicio_carga <= hora <= hora_fin_carga:
        pass
    else:
        model += (
            C_t[t] == 0,
            f"Charge_Hours_{t}"
        )
        model += (
            charge_status[t] == 0,
            f"Charge_Status_Hours_{t}"
        )

    # Horas de descarga: 7 PM a 6 AM
    if hora_inicio_descarga <= hora or hora <= hora_fin_descarga:
        pass
    else:
        model += (
            D_t[t] == 0,
            f"Discharge_Hours_{t}"
        )
        model += (
            discharge_status[t] == 0,
            f"Discharge_Status_Hours_{t}"
        )

solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print(pulp.LpStatus[model.status])

# Si la solución es óptima, puedes extraer y analizar los resultados
if pulp.LpStatus[model.status] == 'Optimal':
    C_sol = [C_t[t].varValue for t in t_indices]
    D_sol = [D_t[t].varValue for t in t_indices]
    SOC_sol = [SOC_t[t].varValue for t in t_indices]

    results = pd.DataFrame({
        'Hora': [t for t in t_indices],
        'Costos_Marginales': [costos_marginales_enero[t] for t in t_indices],
        'Generacion_PV': [generacion_pv[t] for t in t_indices],
        'Carga_BESS': C_sol,
        'Descarga_BESS': D_sol,
        'SOC': SOC_sol,
    })

    # Mostrar los resultados para la primera semana
    print(results.head(48))
else:
    print('La optimización no fue exitosa. Estado:', pulp.LpStatus[model.status])



### Ahora, vamos a graficar los resultados

# 1. Calculo del despacho de energía neto
results['Despacho_neto'] = results['Generacion_PV'] - results['Carga_BESS'] + results['Descarga_BESS']

# # vamos a graficar el despacho neto de energía de un día
# plt.figure(figsize=(12, 6))
# plt.plot(results['Hora'][0:48], results['Despacho_neto'][0:48], label='Despacho neto a la red')
# plt.legend()
# plt.xlabel('Hora')
# plt.ylabel('Potencia (MW)')
# plt.title('Despacho de energía neto')
# plt.show()

# plt.figure(figsize=(15, 7))
# plt.bar(results['Hora'][0:48], results['Carga_BESS'][0:48], label='Carga BESS (MW)', color='blue', alpha=0.5)
# plt.bar(results['Hora'][0:48], results['Descarga_BESS'][0:48], label='Descarga BESS (MW)', color='red', alpha=0.5)
# plt.xlabel('Hora')
# plt.ylabel('Potencia (MW)')
# plt.title('Curvas de Carga y Descarga del BESS - Enero')
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(15, 7))
# plt.plot(results['Hora'][0:48], results['SOC'][0:48], label='Estado de Carga BESS (MWh)', color='purple')
# plt.xlabel('Hora')
# plt.ylabel('Energía Almacenada (MWh)')
# plt.title('Estado de Carga del BESS - Enero')
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(15, 7))

# # Generación PV
# plt.plot(results['Hora'][0:48], results['Generacion_PV'][0:48], label='Generación PV (MW)', color='gold')

# # Despacho Neto al Grid
# plt.plot(results['Hora'][0:48], results['Despacho_neto'][0:48], label='Despacho Neto al Grid (MW)', color='green')

# # Carga BESS
# plt.bar(results['Hora'][0:48], results['Carga_BESS'][0:48], label='Carga BESS (MW)', color='blue', alpha=0.5)

# # Descarga BESS
# plt.bar(results['Hora'][0:48], results['Descarga_BESS'][0:48], label='Descarga BESS (MW)', color='red', alpha=0.5)

# plt.xlabel('Hora')
# plt.ylabel('Potencia (MW)')
# plt.title('Operación de la Planta PV y BESS - Enero')
# plt.legend()
# plt.grid(True)
# plt.show()


# fig, ax1 = plt.subplots(figsize=(15, 7))

# color = 'tab:orange'
# ax1.set_xlabel('Hora')
# ax1.set_ylabel('Costos Marginales (USD/MWh)', color=color)
# ax1.plot(results['Hora'][0:48], results['Costos_Marginales'][0:48], color=color, label='Costos Marginales')
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # Compartir el eje X

# color = 'tab:blue'
# ax2.set_ylabel('Potencia (MW)', color=color)
# ax2.bar(results['Hora'][0:48], results['Carga_BESS'][0:48], label='Carga BESS (MW)', color='blue', alpha=0.5)
# ax2.bar(results['Hora'][0:48], results['Descarga_BESS'][0:48], label='Descarga BESS (MW)', color='red', alpha=0.5)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()
# plt.title('Costos Marginales y Operación del BESS - Enero')
# fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
# plt.show()



# Añadir columnas de 'Día' y 'Hora'
results['Día'] = results['Hora'] // 24 + 1  # Días de 1 a 31
results['Hora_del_día'] = results['Hora'] % 24  # Horas de 0 a 23


# Crear una columna combinada de 'Día-Hora' para etiquetas
results['Día-Hora'] = results['Día'].astype(str) + '-' + results['Hora_del_día'].astype(str)

# Graficar para varios días (ejemplo: días 1 a 3)
dias_a_graficar = [1, 2, 3, 4, 5, 6, 7]
datos_dias = results[results['Día'].isin(dias_a_graficar)]

plt.figure(figsize=(15, 7))
plt.plot(datos_dias.index, datos_dias['Generacion_PV'], label='Generación PV (MW)', color='gold')
plt.plot(datos_dias.index, datos_dias['Despacho_neto'], label='Despacho Neto al Grid (MW)', color='green')
plt.bar(datos_dias.index, datos_dias['Carga_BESS'], label='Carga BESS (MW)', color='blue', alpha=0.5)
plt.bar(datos_dias.index, datos_dias['Descarga_BESS'], label='Descarga BESS (MW)', color='red', alpha=0.5)
plt.xticks(rotation=45)

plt.xlabel('Hora del Día')
plt.ylabel('Potencia (MW)')
plt.title(f'Operación de la Planta PV y BESS - Días {dias_a_graficar}')
plt.legend()
plt.grid(True)

# Ajustar las etiquetas del eje X
num_puntos = len(datos_dias)
etiquetas = datos_dias['Hora_del_día']
ubicaciones = range(len(datos_dias))

plt.xticks(ticks=ubicaciones, labels=etiquetas)
plt.show()
