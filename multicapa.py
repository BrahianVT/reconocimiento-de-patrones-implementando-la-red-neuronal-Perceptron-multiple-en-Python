import numpy as np
import os,cv2,random
'''
Desarrollado por :Brahian Velazquez Tellez
Requerimientos : Python 2.7
cualquier version de Numpy
cualquier version de OpenCv

'''
class perceptron_multipaca:
	largo_patron_entrada = 17
	ancho_patron_entrada = 14
	bias = 1 
	def __init__(self, nombre_carpeta):
		self.nombre_carpeta = nombre_carpeta
	def subir_patrones_entrenamiento(self):
		#sube todos los patrones de entrenamiento y crea la matriz de pesos
		directorio = os.listdir(self.nombre_carpeta)
		self.directorio = directorio
		vectores_entrenamiento = np.empty((len(directorio),(self.largo_patron_entrada * self.ancho_patron_entrada) + 1), dtype = int )
		vectores_entrenamiento[:,0] = self.bias
		for i, imagen in enumerate(directorio):
			ruta_imagenes = os.path.join(self.nombre_carpeta, imagen)
			imagenes = cv2.imread(ruta_imagenes,0)
			vectores_entrenamiento[i][1:] = imagenes.reshape(1,self.largo_patron_entrada * self.ancho_patron_entrada)
			vectores_entrenamiento[vectores_entrenamiento == 0] = -1
			vectores_entrenamiento[vectores_entrenamiento == 255] = 1
		return vectores_entrenamiento
	def realizar_perceptron_multicapa(self):
		#metodo principal que realiza el procedimiento de la red 
		if os.path.exists('vectores_entrenamiento.npy') :
			vectores_entrenamiento = np.load('vectores_entrenamiento.npy')
		else: 
			vectores_entrenamiento = self.subir_patrones_entrenamiento()
			np.save('vectores_entrenamiento.npy', vectores_entrenamiento)
		matriz_pesos = np.random.randn(vectores_entrenamiento.shape[0] , vectores_entrenamiento.shape[1])
		salidas = np.ones((vectores_entrenamiento.shape[0],vectores_entrenamiento.shape[0]), dtype = float)
		salidas*=-1
		for i in range(vectores_entrenamiento.shape[0]):
			salidas[i][i] = 1
		entrada = np.empty((1,vectores_entrenamiento.shape[1]),dtype= float)
		parar = True
		cont = 1
		numero_entrada = 0
		promedio_cero = []
		contar_ceros = 0
		alfa = 1
		salidas2 = np.empty((vectores_entrenamiento.shape[0],1),dtype=float) 
		while parar:
			entrada[0,:] = vectores_entrenamiento[numero_entrada,:]
			multiplicar = np.dot(matriz_pesos,np.transpose(entrada))
			multiplicar[multiplicar >=0.0] = 1
			multiplicar[multiplicar < 0.0] = -1
			salidas2[:,0]= salidas[:,numero_entrada]
			error = salidas2 - multiplicar
			error_cuadratico = np.sum((salidas2 - multiplicar)**2)
			error_cuadratico/= salidas.shape[0]
			if error_cuadratico != 0.0:
				promedio_cero.append(error_cuadratico/float(numero_entrada+1))
				contar_ceros = 0
				numero_entrada = 0
				matriz_pesos = self.nuevos_pesos(matriz_pesos, error,entrada , alfa)
				cont+=1
			else:
				numero_entrada+=1
				contar_ceros+=1
			if cont == 100000 or contar_ceros == vectores_entrenamiento.shape[0]:
				print "Iteraciones realizadas : ",cont
				promedio_cero.append(0.0)
				break
		np.save('matriz_final',matriz_pesos)
		return matriz_pesos
	def nuevos_pesos(self,matriz_pesos, error,entrada , alfa):
		# metodo para encontrar los nuevos pesos de la matriz de prueba 
		return matriz_pesos + (alfa* np.dot(error,entrada))
				
	def subir_patron_de_prueba(self, nombre_patron):
		# metodo para subir el patron de prueba y adicionarle ruido
		ver = 'clase 1' + '/' + nombre_patron
		print ver
		patron_prueba = cv2.imread(ver, 0)
		cv2.imwrite('patron_entrada.bmp',patron_prueba)
		print "arregando ruido al patron de prueba..."
		#24,70,119
		for i in range(119):
			i2 = random.randint(0, 16)
			j = random.randint(0, 13)
			if patron_prueba[i2][j] == 0:
				patron_prueba[i2][j] = 255
			else:
				patron_prueba[i2][j]= 0
		cv2.imwrite('patron_con_ruido.bmp', patron_prueba)
		patron_prueba = patron_prueba.astype('int')
		patron_prueba[patron_prueba == 255] = 1
		patron_prueba[patron_prueba == 0] = -1
		agregando_bias = np.empty((1,(patron_prueba.shape[0] * patron_prueba.shape[1] + 1)), dtype= int)
		agregando_bias[0][0] = 1  
		agregando_bias[0][1:] = patron_prueba.reshape(1 , patron_prueba.shape[0] * patron_prueba.shape[1])
		return agregando_bias
		
	def probar_patron_de_prueba(self):
		#metodo para subir un patron de prueba de los mismos que se tomaron paraentrenar la red 
		#################  Aqui cambia el patron de prueba de entrada con el que quieras que este en la carpeta de clase 1 #######################
		patron_prueba = self.subir_patron_de_prueba('b.bmp'
		if os.path.exists('matriz_final.npy') :
			matriz_pesos = np.load('matriz_final.npy')
		else: 
			matriz_pesos = self.realizar_perceptron_multicapa()
			np.save('matriz_final.npy', matriz_pesos)
		
		ver_resultado = np.dot(matriz_pesos,np.transpose(patron_prueba))
		maximo = max(ver_resultado)
		for i in range(ver_resultado.shape[0]):
			if maximo == ver_resultado[i][0]:
				break
		self.patron_encontrado(i)
		ver_resultado[ver_resultado == maximo] = 1
		ver_resultado[ver_resultado!= maximo] =-1
		print "Etiquetas de salida encontrada"
		print ver_resultado
	
	def patron_encontrado(self, maximo):
		#metodo para guardar el patron encontrado se guardara en forma de imagen.bmp 
		patrones = np.load('vectores_entrenamiento.npy')
		patron_encontrado = patrones[maximo,1:]
		patron_encontrado[patron_encontrado == 1] = 255
		patron_encontrado[patron_encontrado == -1] = 0
		cv2.imwrite('patron_encontrado.bmp',(patron_encontrado.reshape(self.largo_patron_entrada, self.ancho_patron_entrada)))
				
probar = perceptron_multipaca('clase 1')
probar.probar_patron_de_prueba()