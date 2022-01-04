# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:37:47 2021

@author: dtrelles, dortuño
"""


import sys
sys.path.insert(1, r'C:\Users\dtrelles\Documents\GitHub\Sapphire_HTTP\code')
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd


def RegresionLinealSimple(I,V):
    if type(I) is not np.ndarray:
        I = I.to_numpy()
        V = V.to_numpy()
    
    x = I.reshape(-1,1).astype('complex128')
    y = V.reshape(-1,1).astype('complex128')
        
    # if (type(x[0,0]) != np.complex128) and (type(x[0,0]) != complex):
    #     linreg = LinearRegression().fit(x,y)
    #     R2 = linreg.score(x,y)
    #     Z = linreg.coef_[0][0]
    #     C = linreg.intercept_[0]
    # else:
    x = np.concatenate((x,np.ones((len(x),1), dtype='complex128' )*(1+1j)), axis = 1)
    X = np.asmatrix(x, dtype='complex128')
    Reg = np.linalg.inv(X.H@X)@(X.H@y)
    Z = Reg.item(0)
    C = Reg.item(1)
    f = (x[:,0]*Z+C).reshape(-1,1)
    Sreg = sum(abs(y-f)**2)
    Savg = sum(abs(y-np.mean(y))**2)
    R2 = 1 - (Sreg/Savg)[0]
    
    return R2, Z, C

def RegresionLinealR2(I,V, limR2 = 0.9, n = 10):
    # Metodo basado en el paper 'Utility Harmonic Impedance Measurement Based on Data Selection'
    #Divide la medicion en subgrupos de n longitud y toma solamente los subgrupos que tuvieron una R2 mayor a limR2 en su regresion
    x = I.to_numpy().reshape(-1,1)
    y = V.to_numpy().reshape(-1,1)
    
    xsplit = np.array_split(x,np.floor(len(x)/n))
    ysplit = np.array_split(y,np.floor(len(y)/n))
    
    R2 = np.zeros(len(xsplit))
    Z = np.zeros(len(xsplit), dtype = 'complex128')
    C = np.zeros(len(xsplit),dtype = 'complex128')
    
    for i in range(len(xsplit)):
        R2[i], Z[i], C[i] = RegresionLinealSimple(xsplit[i],ysplit[i])
    
    idx = [i for i,j in enumerate(R2) if j >= limR2]
    
    if len(idx) > len(xsplit)*.05:
        return np.mean(R2[idx]), np.mean(Z[idx]), np.mean(C[idx])
    else:
        return 0,Z[0], C[0] 

def RegresionLinealVAR(I,V, n = 10):
     
    # Metodo basado en el paper 'Utility Harmonic Impedance Measurement Based on Data Selection'.
    # Divide la medicion en subgrupos de n longitud y checa las varianzas de cada uno. Toma el 10% de los subgrupos con var(Ipcc) mas alta. 
    #   Esto funciona pq Iu e Ic son independientes(ish), por lo que cuando hay mayor varianza en Ipcc es pq una de las dos fuentes esta mas voltatil,
    #   lo que hace que la otra sea 'constante' en comparacion.
    
    #Al final, checa que la varianza entre las impedancias estimadas con los subgrupos seleccionados sea menos del 20% de la varianza 
    #   entre las impedancias obtenidas con TODOS los subgrupos. Esto lo hace para verificar que los subgrupos seleccionados sean consistentes 
    #   entre si (comparado contra todos los datos)
    
    
        
    x = I.to_numpy().reshape(-1,1)
    y = V.to_numpy().reshape(-1,1)
    x = x[0:len(x) - len(x)%n] #Se cortan los ultimos datos para tener un multiplo de n
    y = y[0:len(y) - len(y)%n]
    
    xsplit = x.reshape(-1,n) #Lo separe por subgrupos de n datos (axis 0 = samples; axis 1 = subgrupos)
    ysplit = y.reshape(-1,n)
    
    varX = np.var(abs(xsplit), axis = 1)   #Vector con la varianza de cada subgrupo
    varLim = np.percentile(varX,90)   #90 percentil de las varianzas, solo se toman los subgrupos con una varianza mayor a esa
    varIdx = varX>=varLim             #Indice de los subgrupos con varianza alta

    R2 = np.zeros(len(xsplit))
    Z = np.zeros(len(xsplit), dtype = 'complex128')
    C = np.zeros(len(xsplit), dtype = 'complex128')
    for i in range(len(xsplit)):  #Se tiene que sacar el de todos para comparar las varianzas de Z al final
        R2[i], Z[i], C[i] = RegresionLinealSimple(xsplit[i],ysplit[i])
    
    
    varSubgroups = np.var(Z[varIdx]) #Varianza de las impedancias obtenidas con los subgrupos de varianza altos
    varTotal = np.var(Z) #Varianza de las impedancias obtenidas con todos los datos
    
    
    if varSubgroups/varTotal < 0.2: #Si se pasa el filtro, se devuelven los promedios de los resultados obtenidos con los subgrupos selectos
       
        return np.mean(R2[varIdx]), np.mean(Z[varIdx]), np.mean(C[varIdx])
        
    else:
        return 0,0,0
   
     
# Los ultimos dos a fiinal de cuentas se terminaror usando como metodos en la clase Estimator pero pues aqui los dejo por si se llegan a usar

def Separacion_de_modos(I,V, eps = 0.3):
    #Metodo para separar en modos. Los modos son bastante subjetivos, dependen del objetivo para el que quieras separar modos.
    #En este caso el objetivo es que se tengan Zh consistentes en cada modo, y que se cambie de modo pocas veces en el dia.
    # Por el segundo motivo se pone Z(t-1) y Z(t+1); para que el algoritmo tome en cuenta el estado justo antes  y justo despues del instante a analizarse.


    idx = (I !=0) & (V!=0)    #Te quita los valores en los que alguno de los 2 es zero
    
    #parametros
    x1 = I[idx].to_numpy().reshape(-1,1)/max(I)         #Ih
    x2 = V[idx].to_numpy().reshape(-1,1)/max(V)         #Vh
    x3 = x2/x1                                          #Zh
    x4 = np.append(x3[1:,:],x3[0,:]).reshape(-1,1)      #Zh(t-1); Zh en la medicion previa
    x5 = np.append(x3[-1,:],x3[0:-1,:]).reshape(-1,1)   #Zh(t+1); Zh en la medicion posterior
    
    X = np.concatenate((x1,x2,x3, x4, x5), axis = 1)
    
    samp = int(len(X)/50)                               #Para que se considere un modo, deben de contener por lo menos el 2% de los datos
    c = DBSCAN(eps = eps, min_samples = samp).fit_predict(X)    #El eps es bastante subjetivo, depende mucho de los parametros de entrada
    mod = pd.Series(index = idx[idx==True].index, data = c) 
    return mod #el index te dice cuales datos se borraron   

def criticalImpedance(Vpcc, Ipcc, Zu, Zc):
    # Esta función está diseñada para implementar el método de impedancia
    # crítica de Wilsun Xu, Chun Li y Tayjasanant T. Este método sirve para 
    # determinar el mayor responsable de la contaminación armónica en 
    # corriente en el punto de conexion.
    #
    # Inputs:
    #   
    #   Vpcc - medición única de voltaje en el punto de conexión (con angulos)
    #   Ipcc - medición única de voltaje en el punto de conexión (con angulos)
    #   Zu - valor estimado de la impedancia de la red (A partir de los metodos)
    #   Zc_min - valor mínimo estimado de la impedancia del cliente
    #   Zc_max - valor máximo estimado de la impedancia del cliente
    #
    # Outputs:
    #
    #   resp - Determina la responsabilidad de la contaminación armónica. Si
    #           'c'  -> el cliente es el mayor responsable
    #           'u'  -> la red es la mayor responsable
    #           'not conlusive'  -> el método no es conclusivo
    #   IET - (impedance error tolerance index) índice que determina si la 
    #           confiabilidad del resultado. Ej. Si IET = 200% quiere decir
    #           que el resultado es confiable incluso si se tuvo un error de 
    #           200% en la Z
    #   CI - Valor de la impedancia crítica
    
    # Definir Z máxima, mínima y la Z del sistema    
    if type(Zc) is pd.Series:   #Si Zc es la que te arrojo la regresion, va a salir un promedio. Si no, va a devolver todo el array, por eso hay que diferenciar para los dos casos
        Zc_min = Zc[abs(Zc).idxmin()]
        Zc_max = Zc[abs(Zc).idxmax()]
    else:
        Zc_min = Zc * 0.5
        Zc_max = Zc * 2
    if type(Zu) is not complex:     #En teoria siempre deberia ser complex, pero ps por si acaso luego cambio algo
        Zu = Zu * 1j
        
    Z = Zu + (Zc_max + Zc_min)/2
    
    
    Zmax = Zu + Zc_max
    Zmin = Zu + Zc_min
    
    # Se define la corriente con el signo negativo por convención del paper
    I = 0-Ipcc
    
    # Fuente de voltaje de la red (Page 674 in paper/ Methodology)
    Eu = Vpcc+Ipcc*Zu
    
    # Ángulo entre el fasor de Eu y la corriente en el punto de conexión (Fig3)
    theta = np.angle(Eu)-np.angle(I)
    
    # ángulo de corrección al método por ángulo en Z (After Eq. 14)
    if np.imag(Z) != 0:
        beta = np.arctan(np.real(Z)/np.imag(Z))
    else:
        beta = np.pi/2
            
    
    # Valor de impedancia crítica (Page 674 in paper/ Methodology)
    CI = 2*np.abs(Eu/I)*np.sin(theta+beta)  # Critical impedance
    
    # impedance error tolerance index (Eq. 13)
    IET = np.abs((np.abs(CI)-np.abs(Z))/Z)
    
    # Si la CI > 0, la culpa es del cliente
    if CI > 0:      
        resp = 'c'
    else: # Si CI < 0
        # Si la magnitud de la CI es mayor que la de la Zmax: culpa de la red
        if np.abs(CI) > np.abs(Zmax):
            resp = 'u'
        
        # Si la magnitud de la CI es menor que la de la Zmin: culpa del cliente
        elif np.abs(CI) < np.abs(Zmin):
            resp = 'c'
            
        # Si CI está fuera de rango, no es conclusivo el test
        else:
            resp = 'not conclusive'
            
    return resp, IET, CI
    