import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pandas as pd


def grafico_curva_despacho(results):
    # Seleccionar un período representativo (por ejemplo, una semana)
    # Aquí seleccionaremos la primera semana
    horas_por_dia = 24
    dias_a_mostrar = 7
    horas_a_mostrar = horas_por_dia * dias_a_mostrar

    datos_grafico = results.iloc[:horas_a_mostrar]

    plt.figure(figsize=(15, 7))
    plt.plot(datos_grafico['Hora'], datos_grafico['Generacion_PV'], label='Generación PV (MW)', color='gold')
    plt.bar(datos_grafico['Hora'], datos_grafico['Carga_BESS'], label='Carga BESS (MW)', color='blue', alpha=0.5)
    plt.bar(datos_grafico['Hora'], -datos_grafico['Descarga_BESS'], label='Descarga BESS (MW)', color='red', alpha=0.5)
    plt.plot(datos_grafico['Hora'], datos_grafico['Despacho_Neto_Grid'], label='Despacho Neto al Grid (MW)', color='green')

    plt.xlabel('Hora')
    plt.ylabel('Potencia (MW)')
    plt.title('Curva de Despacho de Energía - Primera Semana')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def mapa_calor_cmg(results):
    # Calcular el costo marginal promedio por hora y mes
    pivot_cmg = results.groupby(['Mes', 'Hora_del_día'])['CMg'].mean().reset_index()
    pivot_cmg = pivot_cmg.pivot(index='Hora_del_día', columns='Mes', values='CMg')

    # Crear el mapa de calor
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_cmg, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
    plt.title('Mapa de Calor de Costos Marginales Promedio por Hora y Mes')
    plt.xlabel('Mes')
    plt.ylabel('Hora del Día')
    plt.show()

def perfil_operacion_bateria(results):
    # Seleccionar un período representativo (por ejemplo, una semana)
    # Aquí seleccionaremos la primera semana
    horas_por_dia = 24
    dias_a_mostrar = 7
    horas_a_mostrar = horas_por_dia * dias_a_mostrar

    datos_grafico = results.iloc[:horas_a_mostrar]

    plt.figure(figsize=(15, 7))
    plt.plot(datos_grafico['Hora'], datos_grafico['SOC'], label='Estado de Carga del BESS (MWh)', color='purple')

    plt.xlabel('Hora')
    plt.ylabel('Estado de Carga (MWh)')
    plt.title('Perfil de Operación del BESS - Primera Semana')
    plt.legend()
    plt.grid(True)
    plt.show()

def utilizacion_bess_mensual(results):
    # calcular utilizacion del bess mensual
    dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    energia_descargada_mes = results.groupby('Mes')['Descarga_BESS'].sum().reset_index(name='Energia_Descargada')
    # Añadir el número de días por mes al DataFrame
    energia_descargada_mes['Dias_Mes'] = dias_por_mes

    # Capacidad del BESS en MWh (asegúrate de que 'bess_initial_energy_capacity' está definido)
    bess_capacity = 36 # si no funciona, chantar 36

    # Calcular la capacidad total de descarga en cada mes
    energia_descargada_mes['Capacidad_Total_Mes'] = bess_capacity * energia_descargada_mes['Dias_Mes']

    # Calcular la utilización del BESS en cada mes
    energia_descargada_mes['Porcentaje_Utilizacion'] = (
        energia_descargada_mes['Energia_Descargada'] / energia_descargada_mes['Capacidad_Total_Mes']
    ) * 100

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(energia_descargada_mes['Mes'], energia_descargada_mes['Porcentaje_Utilizacion'], color='skyblue')

    # Personalizar el gráfico
    plt.xlabel('Mes')
    plt.ylabel('Porcentaje de Utilización del BESS (%)')
    plt.title('Utilización Mensual del BESS en Descarga')
    plt.xticks(range(1, 13))  # Mostrar los meses del 1 al 12
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def grafico_curva_despacho_mes(results, mes):
    # Filtrar los datos para el mes especificado
    datos_mes = results[results['Mes'] == mes]

    plt.figure(figsize=(15, 7))
    plt.plot(datos_mes['Hora'], datos_mes['Generacion_PV'], label='Generación PV (MW)', color='gold')
    plt.bar(datos_mes['Hora'], datos_mes['Carga_BESS'], label='Carga BESS (MW)', color='blue', alpha=0.5)
    plt.bar(datos_mes['Hora'], -datos_mes['Descarga_BESS'], label='Descarga BESS (MW)', color='red', alpha=0.5)
    plt.plot(datos_mes['Hora'], datos_mes['Despacho_Neto_Grid'], label='Despacho Neto al Grid (MW)', color='green')

    plt.xlabel('Hora')
    plt.ylabel('Potencia (MW)')
    plt.title(f'Curva de Despacho de Energía - Mes {mes}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def menu_graficos(results):
    graficos = True
    while graficos:
        print("Seleccione una opción:")
        print("1. Graficar la curva de despacho de energía")
        print("2. Mostrar el mapa de calor de costos marginales")
        print("3. Mostrar el perfil de operación de la batería")
        print("4. Mostrar la utilización mensual del BESS")
        print("5. Mostrar Grafico Curva Despacho de un Mes Especifico")
        print("6. Salir")
        opcion = input("Opción: ")

        if opcion == '1':
            grafico_curva_despacho(results)
        elif opcion == '2':
            mapa_calor_cmg(results)
        elif opcion == '3':
            perfil_operacion_bateria(results)
        elif opcion == '4':
            utilizacion_bess_mensual(results)
        elif opcion == '5':
            mes = int(input("Ingrese el mes que desea visualizar (1-12): "))
            grafico_curva_despacho_mes(results, mes)
        elif opcion == '6':
            graficos = False
        
        
