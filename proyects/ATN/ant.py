import random as rn
import numpy as np
from numpy.random import choice as np_choice
from math import sqrt
import matplotlib.pyplot as plt
import json
import pandas as pd

#from google.colab import files
#uploaded = files.upload()
#with open("Datos_ANT_Ingreso_de_nodos.json", "r") as tsp_data:
#    tsp = json.load(tsp_data)

with open("proyects/ATN/Datos_ANT_Ingreso_de_nodos.json", "r") as tsp_data:
    tsp = json.load(tsp_data)


distances0 = tsp["DistanceMatrix"]
tour_size=tsp["TourSize"]
for i in range(tour_size):
  distances0[i][i]=np.inf
distances0=np.array(distances0)
#Convertimos la matriz de distancias 
#en DataFrame para manipular facilmente
df = distances0.copy() #Creamos una copia profunda para 
#no modificar el arreglo original
df = pd.DataFrame(df)
nodos_totales= df.columns.tolist()#Almacenamos todos los 
#nodos de la matriz como una lista
nodos_totales.remove(0)#Debemos remover el 0, ya que es 
#un nodo que siempre vamos a visitar
rang_min = nodos_totales[0]
rang_max = nodos_totales[-1]

#Ahora Vamos a pedir los nodos por los cuales queremos pasar
nodos=[]
#print("Ingresar los nodos que desea visitar, 
#\ncuando los complete ingrese 0 para finalizar:\n")
while True:
  valor=int(input(f'La lista de nodos a recorrer es -> {nodos} \nInserte el nodo que desea agregar a la lista o inserte 0 para finalizar: '))
  if valor != 0:
    if valor > rang_max or valor<rang_min:
      print('Debe ingresar un número dentro del rango')
    else:
      if valor in nodos:
        print('Este nodo ya fue seleccionado')
      else:
        nodos.append(valor)
        nodos.sort()
  else:
    break

#Con el siguiente ciclo for comparamos la matriz de todos los 
#nodos que tiene la matriz, con los que queremos visitar
#para dejar los que no queremos visitar y volverlos infinitos 
for i in nodos:
  if i in nodos_totales:
    nodos_totales.remove(i)

#Con la función iloc, hacemos que los nodos que no se eligieron 
#visitar las filas y columnas tomen el valor infinito
df.iloc[:,nodos_totales] = np.inf
df.iloc[nodos_totales] = np.inf
distances = np.array(df)
pheromone = np.ones(distances.shape) / len(distances)
pheromone = pd.DataFrame(pheromone)
pheromone.iloc[:,nodos_totales] = 0
pheromone.iloc[nodos_totales] = 0
pheromone = np.array(pheromone)
cantidad_no_visitados = len(nodos_totales)

class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is- 
            -multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(distances, 100, 20, 2000, 0.95, alpha=1, beta=2)          
        """
        self.distances  = distances
#       self.pheromone = np.ones(self.distances.shape) / len(distances) 
        #al principio cada arco se marca con la misma cantidad de feromonas
        self.pheromone = pheromone
        self.all_inds = range(len(distances)) #generamos una lista de nodos a visitar
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha 
        self.beta = beta
        self.cantidad_no_visitados = cantidad_no_visitados

    def run(self):
        distance_logs=[]
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf) #inicialmente no hay ruta más corta, 
        #la longitud más corta para pasar por todos los nodos es +infinito
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths() #generamos los caminos de cada hormiga
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)  
            #la matriz de feromonas se modifica: las hormigas depositan las 
            #feromonas por donde han pasado
            shortest_path = min(all_paths, key=lambda x: x[1])  #seleccionamos el camino 
            #más corto: según la longitud del camino (el segundo elemento de la tupla)
            if shortest_path[1] < all_time_shortest_path[1]:  #si encontramos 
            #una ruta más corta que all_time_shortest_path, modificamos all_time_shortest_path
                all_time_shortest_path = shortest_path
            distance_logs.append(all_time_shortest_path[1]) 
            self.pheromone * self.decay #agregado, nuevo, decadencia de las fermonas                     
        return all_time_shortest_path,distance_logs

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])  #ordenamos los caminos 
        #según su longitud (distancia total, el segundo elemento de la tupla)
        for path, dist in sorted_paths[:n_best]:  #depositaremos feromonas en 
        #los n_best caminos más cortos
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]  #añadimos 
                #feromonas en cada uno de los arcos por donde han pasado las mejores hormigas

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path: # la distancia total recorrida por la hormiga es 
        #la suma de las distancias recorridas en cada uno de los viajes
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants): # para cada una de las n_ants ants, 
        #agregue a la lista all_paths el camino seguido por ant i
            path = self.gen_path(0) # generamos un camino comenzando desde 
            #el primer nodo (0) y pasando por todos los nodos
            all_paths.append((path, self.gen_path_dist(path))) # la ruta seguida
            #por una hormiga tiene la forma de una tupla: el primer elemento es 
            #la lista de nodos visitados, el segundo elemento es la distancia total 
            #recorrida por la hormiga hormiga en este camino
        return all_paths


    def gen_path(self, start):
        path = [] # la lista de movimientos de un nodo a otro, que iremos 
        #llenando a medida que se mueva la hormiga
        visited = set() # un conjunto es una lista desordenada, aquí creamos 
        #un conjunto vacío que corresponde a todos los nodos que la hormiga 
        #ya ha visitado. Esto es para evitar que la hormiga visite el mismo nodo dos veces.
        visited.add(start) # agregamos el nodo inicial al conjunto de nodos visitados
        prev = start #prev es una variable que almacena, cada vez que la hormiga 
        #se mueve, el nodo desde el que parte
        for i in range(len(self.distances) - (1+cantidad_no_visitados)): # para visitar 
        #todos los nodos del grafo una sola vez, la hormiga tendrá que hacer tantos 
        #movimientos como nodos haya -1
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited) 
            # seleccionamos el destino del movimiento realizado por la hormiga 
            #en esta etapa del bucle, que almacenamos en la variable move
            path.append((prev, move)) # agregar el movimiento desde anterior 
            #para pasar a la lista de rutas
            prev = move # el nodo de movimiento ahora es el nodo donde está la hormiga, 
            #por lo que es el punto de partida para el siguiente movimiento
            visited.add(move) # agregamos el nodo donde está la hormiga 
            #al conjunto de nodos visitados
        path.append((prev, start)) # volviendo a donde comenzamos: 
        #el último movimiento va del último nodo al primero, al bucle

        #Aquí termina
        return path
        

    def pick_move(self, pheromone, dist, visited): # tomamos como argumento 
    #la fila de la matriz self.pheromone correspondiente al nodo inicial, 
    #la fila de la matriz self.distances correspondiente a este nodo, 
    #y el conjunto de nodos ya visitado
        pheromone = np.copy(pheromone) # copiamos la lista de feromonas 
        #para actuar sobre ella en la función, sin modificar la línea de 
        #la matriz self.pheromone que es un atributo del objeto 
        pheromone[list(visited)] = 0 # para evitar que la hormiga elija un nodo 
        #que ya ha sido visitado, actuamos como si no hubiera feromonas en 
        #los arcos que van desde el nodo inicial hasta los nodos que están 
        #en el conjunto visitado

#        print('Estoy en el primer paso: ', pheromone)

        row = (pheromone ** self.alpha) * (( 1.0 / dist) ** self.beta)  
        #fila es una lista donde cada elemento es el producto de la intensidad 
        #(cantidad de feromonas) del arco y su visibilidad (inverso de su distancia), 
        #ponderado por alfa y beta
#        print('Esto vale Row: ', row)

        norm_row = row / row.sum() # norm_row es la lista donde dividimos 
        #los coeficientes obtenidos anteriormente por la suma de todos estos 
        #coeficientes, para obtener la probabilidad de que la hormiga elija 
        #el nodo considerado
#        print('Esto vale la norma: ',norm_row)
        
        move = np_choice(self.all_inds, 1, p=norm_row)[0] # elección del nodo s
        #egún la ley que hemos construido
#        print(f'\nYa hice la corrida y elegí el nodo:{move}')

        return move

#A partir de acá se llama a la clase para iniciar la ejecución según los parámetros 
ant_colony = AntColony(distances, 200, 200, 150, 0.90, alpha=1, beta=5)
shortest_path,log = ant_colony.run()
print ("shortest_path: {}".format(shortest_path))
plt.plot(log)
plt.show()


'''
Lo siguiente es para presentar la respuesta 
por nodos, optimizando la respuesta,
se puede modificar a gusto
'''

u = shortest_path
a = u[0]
b=u[1]
lista3 = ['Nodo inicial - ']
for k, i in a:
  lista3.append('Nodo ')
  lista3.append(str(i))
  lista3.append(' - ')
lista3.pop()
resultado= (''.join(lista3))
print(f'''


De acuerdo a la optimización del algoritmo se obtiene la siguiente ruta: 

{resultado}.

Esta opción es la optima para el almacen establecido, esta ruta presenta 
una distancia total de {b} metros recorridos.''')