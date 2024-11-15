#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Ejemplo de dataframe**

# In[3]:


#Ejemplo de dataframe
data = {'Nombre': ['Ana', 'Juan', 'María', 'Pedro'],
        'Edad': [20, 22, 19, 21],
        'Carrera': ['Ingeniería', 'Contabilidad', 'Medicina', 'Informática'],
        'Calificación': [8.5, 9.2, 7.8, 9.5]}
df = pd.DataFrame(data)

print(df)


# **Creación de visualizaciones con los datos**
# 

# In[5]:


# Gráfico de barras

sns.barplot(x='Carrera', y='Calificación', data=df)
plt.title('Calificación promedio por carrera')
plt.xlabel('Carrera')
plt.ylabel('Calificación')
plt.show()


# **Cargar información en base a un conjunto de datos (Excel)**
# 

# In[7]:


df = pd.read_excel('inmuebles.xlsx')


# In[8]:


df.head(5)


# In[9]:


#Cantidad de filas y columnas
df.shape


# In[10]:


#Información de los datos
df.info()


# In[11]:


# Lista de campos de la base de datos
for col in df.columns:
    print(col)


# In[12]:


# Características estadísticas de las columnas seleccionadas
df[['Superficie','Precio Venta','Días para Vender']].describe()


# In[13]:


#Este es un estilo de consulta con el numero que ponga. Lo va a buscar a la base de datos y selecciona toda la fila
df.iloc[874]


# In[14]:


#Busca desde el 18 hasta el 20
df.iloc[18:20]


# In[15]:


#Busca en la columna Vendedor, que este en la fila 500
df['Vendedor'][2000]


# In[16]:


#Muestra aleatoria 
df.sample(3)


# In[17]:


# Promedio
promedio = df['Precio Venta'].mean()
print("Promedio:", promedio)

# Mediana
mediana = df['Precio Venta'].median()
print("Mediana:", mediana)

# Moda
moda = df['Precio Venta'].mode()
print("Moda:", moda)


# In[18]:


#Precio promedio de venta por ciudad: Examina los precios de venta promedio para cada ciudad.
df.groupby('Ciudad')['Precio Venta'].mean().round()


# In[19]:


valor_minimo = df['Precio Venta'].min()
valor_maximo = df['Precio Venta'].max()


# **Grafico de lineas de las ventas durante todos los meses**

# In[21]:


#Grafico de lineas para mostrar las vendas en cada mes

import matplotlib.pyplot as plt

# Crear el gráfico de línea
ax = df.groupby(df['Fecha Venta'].dt.to_period('M')).size().plot(kind='line')

# Obtener los datos de X y Y del gráfico
x_values = ax.get_lines()[0].get_xdata()
y_values = ax.get_lines()[0].get_ydata()

# Agregar etiquetas de datos
for x, y in zip(x_values, y_values):
    plt.annotate(f'{int(y)}', xy=(x, y), xytext=(0, 5), textcoords='offset points', ha='center')
#Color y grosor de la linea
ax = df.groupby(df['Fecha Venta'].dt.to_period('M')).size().plot(kind='line', color='red',linewidth=5)


# Personalizar el gráfico
plt.title('Ventas por Meses')
plt.xlabel('Meses')
plt.ylabel('Número de Ventas')
plt.grid(True)
plt.show()


# **Diagrama circular del estatus del inmueble (Vendida vs En proceso)**

# In[23]:


df['Estatus'].value_counts().plot(kind='pie', 
                                  title='Estatus', 
                                  autopct='%1.1f%%',  # Muestra el porcentaje con 1 decimal
                                  startangle=90,  # Para que el gráfico comience desde arriba
                                  figsize=(5, 5),  # Tamaño del gráfico
                                  wedgeprops={'edgecolor': 'black'})  # Bordes de los sectores


# **Tipo y cantidad de cada inmueble**

# In[25]:


import matplotlib.pyplot as plt
import pandas as pd

# Contar la frecuencia de cada tipo de inmueble
conteo_tipos = df['Tipo'].value_counts()

# Crear el gráfico de barras
conteo_tipos.plot(kind='bar', title='Tipos de inmuebles')

# Agregar etiquetas a las barras
for i in range(len(conteo_tipos)):
    plt.text(i, conteo_tipos.iloc[i], str(conteo_tipos.iloc[i]), ha='center', va='bottom')

# Etiquetas de los ejes
plt.xlabel('Inmuebles')
plt.ylabel('Cantidad de inmuebles')
plt.xticks(rotation=45) #inclinacion de los nombres
# Mostrar el gráfico
plt.show()


# **Cantidad de ventas vs alquiler**

# In[29]:


#Alquiler vs Ventas
df['Operación'].value_counts().plot(kind='pie', 
                                    title='Distribución de las operaciones', 
                                    autopct=lambda p: '{:.0f}'.format(p * sum(df['Operación'].value_counts()) / 100),  # Muestra el valor absoluto
                                    startangle=90, 
                                    figsize=(6, 6), 
                                    wedgeprops={'edgecolor': 'black'})

Este grafico de pie o circular nos muestra un breve resumen de las ventas vs los alquileres. Donde las ventas superan lijeramente 
los que son de alquiler. 
# In[ ]:





# **Grafico de Barras horizontales de vendedores por la cantidad de ventas que realizaron**

# In[33]:


import matplotlib.pyplot as plt


# Contar la frecuencia de cada tipo de inmueble
conteo_tipos = df['Vendedor'].value_counts()

# Crear el gráfico de barras horizontales
conteo_tipos.plot(kind='barh', title='Vendedores')

# Agregar etiquetas a las barras (ahora en el eje horizontal)
for i, v in enumerate(conteo_tipos):
    plt.text(v, i, str(v), va='center')

# Etiquetas de los ejes
plt.xlabel('Cantidad de ventas')
plt.ylabel('Nombre del vendedor')



# Mostrar el gráfico
plt.show()

Todos los vendedores tienen un desempeño bastante cercano, donde el primero hasta el último solo hay una 
diferencia de 45 unidades, esta brecha puede ser por diferentes factores como lo son la experiencia, la red de contactos,
la atención al cliente o las capacidades de negociación. Para mejorar estos aspectos se puede hacer:
* Bridar capacitaciones en tecnicas de ventas, manejo de objeciones y cierre de ventas.
* Juntar los mejores vendedores con aquellos que estan por debajo, soluciones personalizadas e insentivos.
Podemos destacar a Luisa y a Carmen como vendedoras estrellas, ya que sus ventas superan el promedio en comparación con sus compañeros.
# In[34]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Seleccionar las características relevantes
X = df[['Superficie', 'Precio Venta']]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Numero de clusters
num_clusters = 3

# Crear y ajustar el modelo K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_scaled)

# Agregar los labels de los clusters al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualizar los clusters
plt.scatter(df['Superficie'], df['Precio Venta'], c=df['Cluster'], cmap='rainbow')
plt.xlabel('Superficie')
plt.ylabel('Precio Venta')
plt.title('Clustering de Inmuebles con K-means')
plt.show()

El grafico de K-means podemos resaltar tres grupos, con caracteristicas distintas
en cuanto tamaño y precio. Podemos councluirque podemos tener tres nichos como por ejemplo viviendas
de lujo, viviendas con tamaño medio y por ultimo casas pequeñas.
Tenemos una relacion positiva debido a que ambas variables "Superficie" y "Precio Venta" aumentan, no es un
patron exacto pero hay una tendencia
# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns

# Descripción de los datos
print(df['Precio Venta'].describe())

# Configuración de la figura
plt.figure(figsize=(6, 5))

# Trazar el histograma y la curva de densidad
sns.histplot(df['Precio Venta'], color='blue', bins=100, kde=True, stat="density", alpha=0.4)

# Mostrar el gráfico
plt.title('Histograma de Precio Venta')
plt.xlabel('Precio Venta')
plt.ylabel('Densidad')
plt.show()


# In[36]:


propiedades_lentas = df.sort_values(by='Días para Vender', ascending=False)
print(propiedades_lentas.head(15).to_string())


# In[37]:


df['Fecha Venta'] = pd.to_datetime(df['Fecha Venta'])
df['Mes'] = df['Fecha Venta'].dt.to_period('M')  # Agrupar por mes
precios_promedio_mensuales = df.groupby('Mes')['Precio Venta'].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(precios_promedio_mensuales['Mes'].astype(str), precios_promedio_mensuales['Precio Venta'], marker='o')
plt.title('Evolución de los Precios de Venta de Inmuebles por Mes')
plt.xlabel('Mes')
plt.ylabel('Precio Promedio de Venta')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor legibilidad

plt.grid()
plt.tight_layout()  # Ajustar el diseño para que no se superpongan elementos
plt.show()

¿Cómo han evolucionado los precios de venta de los inmuebles en los últimos años? 
Para analizar la evolucion de los precios de las ventas se agrupa los promedios por meses,
esto permite visualizar la evolucion de los precios a lo largo del tiempo.

# In[51]:


df_ventas = df[df['Operación'] == 'Venta']
precios_promedio = df_ventas.groupby(['Ciudad', 'Tipo'])['Precio Venta'].mean().reset_index()
precios_promedio.rename(columns={'Precio Venta': 'PrecioPromedio'}, inplace=True)

# Formatear la columna 'PrecioPromedio'
precios_promedio['PrecioPromedio'] = precios_promedio['PrecioPromedio'].apply(lambda x: f"{x:,.2f}")

# Imprimir el DataFrame
print(precios_promedio)

