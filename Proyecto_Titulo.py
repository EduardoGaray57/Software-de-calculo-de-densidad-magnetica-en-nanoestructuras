import pandas as pd
import pyvista as pv
import numpy as np
import os

def mostrar_menu(opciones):
    print('Seleccione una figura geometrica a trabajar:')
    for clave in sorted(opciones):
        print(f'{clave}) {opciones[clave][0]}')

def leer_opcion(opciones):
    while (a := input('Opcion: ')) not in opciones:
        print('Opcion incorrecta, vuelva a intentarlo.')
    return a

def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()

def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()

def menu_principal():
    opciones = {
        '1': ('Opción Cubico', Cubo),
        '2': ('Opción Cilindro', Cilindro),
        '3': ('Opción Esfera', Esfera),
        '4': ('Salir', salir)
    }
    generar_menu(opciones, '4')

def verificacion_ruta(directorio):
    if not os.path.isdir(directorio):
        print('La ruta proporciona no es un directorio valido')
    else:
        dataframe = []
        return dataframe

def eliminar_filas_con_ceros(df):
    return df[(df['m_x'] != 0) & (df['m_y'] != 0) & (df['m_z'] != 0)]

def cargar_archivos(directorio):
    verif = verificacion_ruta(directorio)
    for archivo in os.listdir(directorio):
        if archivo.endswith('.txt'):
            ruta_archivo = os.path.join(directorio, archivo)
            df = pd.read_csv(ruta_archivo, sep='\\s+', comment='#', skiprows=7,
                            names=['x', 'y', 'z', 'm_x', 'm_y', 'm_z'])
            df = eliminar_filas_con_ceros(df)
            verif.append(df)
    if verif:
        combined_df = pd.concat(verif)
        return combined_df
    else:
        print('No se encontraron archivos .TXT en el directorio')

#Cubo
def Cubo():
    print(solicitud_cubo())
    print('Has elegido la opcion del Cubo')

def solicitud_cubo():
    directorio = input('Por favor ingrese la ruta del directorio donde esten los archivos .TXT: ')
    datos = cargar_archivos(directorio)
    if datos is not None:
        lx = input('Ingrese el tamaño de la celda en el eje X: ')
        ly = input('Ingrese el tamñao de la celda en el eje Y: ')
        lz = input('Ingrese el tamñao de la celda en el eje Z: ')
        LX = input('Ingrese el largo del cubo en el eje X: ')
        LY = input('Ingrese el largo del cubo en el eje Y: ')
        LZ = input('Ingrese el largo del cubo en el eje Z: ')
        resultados = calculo_densidad_carga_magnetica_cubo(
            lx, ly, lz, LX, LY, LZ,
            datos['x'].values, datos['y'].values, datos['z'].values,
            datos['m_x'].values, datos['m_y'].values, datos['m_z'].values
            )        
        export_csv_cubo(resultados)
        generar_mapa_cubo(resultados)
        generar_mapa_cubo_componentes(datos)
    return resultados

def calculo_densidad_carga_magnetica_cubo(lx, ly, lz, LX, LY, LZ, x, y, z, mx, my, mz):
    #Convesion de datos de entrada a arrays de Numpy
    x, y, z, mx, my, mz = map(np.array,[x, y, z, mx, my, mz])
    lx = float(lx)
    ly = float(ly)
    lz = float(lz)
    LX = float(LX)
    LY = float(LY)
    LZ = float(LZ)

    resultados = pd.DataFrame({'x': x, 'y': y, 'z': z})

    #Calculo de densidad por cara
    condiciones = {
        'Cara_X+': (np.isclose(x, LX-lx/2, y==0, z==0, ), mx),
        'Cara_X-': (np.isclose(x, lx/2, y==0, z==0), -mx),
        'Cara_Y+': (np.isclose(y, LY-ly/2, x==0,z==0), my),
        'Cara_Y-': (np.isclose(y, ly/2, x==0,z==0), -my),
        'Cara_Z+': (np.isclose(z, LZ-lz/2, x==0,y==0), mz),
        'Cara_Z-': (np.isclose(z, lz/2, x==0,y==0), -mz),
    }
    for key, (mascara, componente) in condiciones.items():
        resultados[key] = np.where(mascara, componente, 0)
    
    return resultados

def export_csv_cubo(df):
    df.to_csv('Data_Cubo.csv', index=False, sep=';')

def generar_mapa_cubo(df):
    plotter = pv.Plotter()
    for cara in ['Cara_X+', 'Cara_X-', 'Cara_Y+', 'Cara_Y-', 'Cara_Z+', 'Cara_Z-']:
        valid_data = df[df[cara] != 0]

        points = valid_data[['x', 'y', 'z']].values
        values = valid_data[cara].values 

        mesh = pv.PolyData(points)
        mesh[cara] = values

        # Ajustar la escala de colores
        plotter.add_mesh(mesh, scalars=cara, point_size=10, render_points_as_spheres=True, cmap="plasma",lighting=True, clim=(-1, 1),scalar_bar_args={'title': "Densidad de carga magnetica"})

    plotter.show()

def generar_mapa_cubo_componentes(df):
    # Crear un objeto Plotter con 3 subplots horizontales
    plotter = pv.Plotter(shape=(1, 3))

    # Diccionario para componentes y títulos
    componentes = {
        'm_x': "Componente Mx",
        'm_y': "Componente My",
        'm_z': "Componente Mz"
    }

    # Iterar sobre cada componente y agregar un subplot
    for idx, (componente, titulo) in enumerate(componentes.items()):
        plotter.subplot(0, idx)
        plotter.add_text(titulo, font_size=10)

        # Filtrar datos no nulos
        valid_data = df[df[componente].notna()]

        # Extraer coordenadas y valores de la cara
        points = valid_data[['x', 'y', 'z']].values
        values = valid_data[componente].values

        # Crear PolyData y asignar valores escalares
        mesh = pv.PolyData(points)
        mesh[componente] = values

        # Agregar la malla al subplot correspondiente
        plotter.add_mesh(
            mesh, 
            scalars=componente, 
            point_size=10, 
            render_points_as_spheres=True, 
            cmap= "plasma", 
            lighting=True, 
            clim=(-1, 1),
            scalar_bar_args={'title': f"Densidad de {titulo}"}
        )

    # Sincronizar las vistas entre subplots
    plotter.link_views()

    # Mostrar la visualización
    plotter.show()

#Cilindro
def Cilindro():
    print(solicitud_cilindro())
    print('Has elegido la opcion del Cilindro')

def solicitud_cilindro():
    directorio = input('Por favor ingrese la ruta del directorio donde esten los archivos .TXT: ')
    datos = []
    datos = cargar_archivos(directorio)
    resultados = calculo_densidad_carga_magnetica_cilindro(datos)
    export_csv_cilindro(resultados)
    generar_mapa_cilindro(resultados)
    generar_mapa_cilindro_componentes(resultados)
    return resultados

def calculo_densidad_carga_magnetica_cilindro(df):
    lz = input('Ingrese el tamñao de la celda en el eje Z: ')
    lz = float(lz)

    df["Cara_Z+"] = -df["m_z"].where(df["z"] > 0, 0)

    # Cara inferior (-Z): \Gamma = -M_z (cuando z < 0)
    df["Cara_Z-"] = df["m_z"].where(df["z"] == lz / 2, 0)

    # Calcular el ángulo φ en radianes
    df["phi"] = np.arctan2(df["y"], df["x"])  # arctan2 maneja todos los cuadrantes

    # Densidad en la superficie lateral: \Gamma = M_x * cos(φ) + M_y * sin(φ)
    df["Densidad_Lateral"] = (-df["m_x"] * np.cos(df["phi"])) + (-df["m_y"] * np.sin(df["phi"]))

    return df

def export_csv_cilindro(df):
    df.to_csv('Data_Cilindro.csv', index=False, sep=';')

def generar_mapa_cilindro(df):
    # Crear un objeto PolyData para cada cara
    plotter = pv.Plotter()

    for cara in ['Cara_Z-','Cara_Z+','Densidad_Lateral']:
        # Filtrar datos no nulos
        valid_data = df[df[cara].notna()] 
    
        # Extraer coordenadas y valores de la cara
        points = valid_data[['x', 'y', 'z']].values
        values = valid_data[cara].values

        # Crear PolyData y asignar valores escalares
        mesh = pv.PolyData(points)
        mesh[cara] = values

        # Agregar la malla al gráfico con un color distinto por cara
        plotter.add_mesh(
            mesh,
            scalars=cara,
            point_size=10,
            render_points_as_spheres=True,
            cmap="viridis",
            #clim=(-0.255, 1),
            scalar_bar_args={'title': 'densidad de carga magnetica'}
            )

    # Mostrar la gráfica
    plotter.show()

def generar_mapa_cilindro_componentes(df):
    # Crear un objeto Plotter con 3 subplots horizontales
    plotter = pv.Plotter(shape=(1, 3))

    # Diccionario para componentes y títulos
    componentes = {
        'm_x': "Componente Mx",
        'm_y': "Componente My",
        'm_z': "Componente Mz"
    }

    colormaps = ['Reds', 'Blues', 'Greens']  # Colormaps para diferenciar cada componente

    # Iterar sobre cada componente y agregar un subplot
    for idx, (componente, titulo) in enumerate(componentes.items()):
        plotter.subplot(0, idx)
        plotter.add_text(titulo, font_size=10)

        # Filtrar datos no nulos
        valid_data = df[df[componente].notna()]

        # Extraer coordenadas y valores de la cara
        points = valid_data[['x', 'y', 'z']].values
        values = valid_data[componente].values

        # Crear PolyData y asignar valores escalares
        mesh = pv.PolyData(points)
        mesh[componente] = values

        # Agregar la malla al subplot correspondiente
        plotter.add_mesh(
            mesh, 
            scalars=componente, 
            point_size=10, 
            render_points_as_spheres=True, 
            cmap= "plasma", 
            lighting=True, 
            clim=(-1, 1),
            scalar_bar_args={'title': f"Densidad de {titulo}"}
        )

    # Sincronizar las vistas entre subplots
    plotter.link_views()

    # Mostrar la visualización
    plotter.show()


#Esfera
def Esfera():
    print(solicitud_esfera())
    print('Has elegido la opcion de la esfera')

def solicitud_esfera():
    directorio = input("Por favor ingrese la ruta del directorio donde esten los archivos .TXT: ")
    datos = []
    datos = cargar_archivos(directorio)
    resultados = calculo_densidad_carga_magnetica_esfera(datos["m_x"], datos["m_y"],
                                    datos["m_z"], datos["x"], datos["y"], datos["z"])
    export_csv_esfera(resultados)
    generar_mapa_esfera(resultados)
    generar_mapa_esfera_componentes(datos)
    return resultados

def calculo_densidad_carga_magnetica_esfera(m_x, m_y, m_z, x, y, z):    
    # Calculo del radio
    radio = np.sqrt(x**2 + y**2 + z**2)
    
    # Calculo de theta
    theta = np.arccos(z / radio)
    
    # Calculo de phi
    phi = np.arctan2(y, x)

    # Calculo de trigonometría 
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)

    # Calculo de carga_superficial
    carga_superficial = ((sin_theta * cos_phi * m_x) + 
                         (sin_theta * sin_phi * m_y) + 
                         (cos_theta * m_z)) 
    
    # Creación del DataFrame
    data = {
        'x': x,
        'y': y,
        'z': z,
        'm_x': m_x,
        'm_y': m_y,
        'm_z': m_z,
        'carga_superficial': carga_superficial
    }
    df = pd.DataFrame(data)
    return df

def export_csv_esfera(df):
    df.to_csv('Data_Esfera.csv', index=False, sep=';')

def generar_mapa_esfera(df):
    # Extraer las coordenadas y la carga superficial desde el DataFrame
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    carga_superficial = df['carga_superficial'].values
    
    # Crear puntos 3D
    points = np.column_stack((x, y, z))
    
    # Crear nube de puntos con PyVista
    point_cloud = pv.PolyData(points)
    point_cloud['carga_superficial'] = carga_superficial  # Asignar carga superficial a los puntos
    
    # Configurar el graficador
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars='carga_superficial', cmap='plasma', point_size=10, render_points_as_spheres=True,clim=(-1, 1))
    plotter.show()

def generar_mapa_esfera_componentes(df):
    # Crear un objeto Plotter con 3 subplots horizontales
    plotter = pv.Plotter(shape=(1, 3))

    # Diccionario para componentes y títulos
    componentes = {
        'm_x': "Componente Mx",
        'm_y': "Componente My",
        'm_z': "Componente Mz"
    }

    colormaps = ['Reds', 'Blues', 'Greens']  # Colormaps para diferenciar cada componente

    # Iterar sobre cada componente y agregar un subplot
    for idx, (componente, titulo) in enumerate(componentes.items()):
        plotter.subplot(0, idx)
        plotter.add_text(titulo, font_size=10)

        # Filtrar datos no nulos
        valid_data = df[df[componente].notna()]

        # Extraer coordenadas y valores de la cara
        points = valid_data[['x', 'y', 'z']].values
        values = valid_data[componente].values

        # Crear PolyData y asignar valores escalares
        mesh = pv.PolyData(points)
        mesh[componente] = values

        # Agregar la malla al subplot correspondiente
        plotter.add_mesh(
            mesh, 
            scalars=componente, 
            point_size=10, 
            render_points_as_spheres=True, 
            cmap= "plasma", 
            lighting=True, 
            clim=(-1, 1),
            scalar_bar_args={'title': f"Densidad de {titulo}"}
        )

    # Sincronizar las vistas entre subplots
    plotter.link_views()

    # Mostrar la visualización
    plotter.show()

def salir():
    print('Saliendo')

if __name__ == '__main__':
    menu_principal()
