# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:56:23 2021

@author: dtrelles
"""
import pandas as pd
import numpy as np






class Limite:
    def __init__(self):
        # Se definen atributos de acuerdo a limites de Ih y Vh
        self.Ilim = np.zeros((1))
        self.Vlim = np.zeros((1))
        self.index = np.zeros(1)
        

    def deterlim(self, Vplanta, Zrel, cb_horno, LIM_VH):
        '''
        This functions find specific index according to "Vplanta" and "Zrel"
        and then identifies limits for Curent and Voltage based on
        "CFE L0000-45" tables.

        Parameters
        ----------
        Vplanta : int
            Nominal Voltaje of the case.
        Zrel : float
            Zrel of the case.
        ruta : string
            Route where to get img from table in "cdr".
        cb_hornos : bool
            Indicates if the case is related to a increases in harmonic limits.

        Returns
        -------
        None.

        '''
        
        #Tablas de la CFE 0000-45, las columnas son es Zrel , h<11, h<17,h<23,h<35, h<50, DATD
        tabla_baja = [[20, 50, 100, 1000, 1000000], [4,7,10,12,15], [2,3.5,4.5,5.5,7], [1.5,2.5,4,5,6], [0.6,1,1.5,2,2.5], [0.3,0.5,0.7,1,1.4], [5,8,12,15,20]]
        tabla_media =[[20, 50, 100, 1000, 1000000], [2,3.5,5,6,7.5],[1,1.75,2.25,2.75,3.5], [0.75,1.25,2,2.5,3], [0.3,0.5,0.75,1,1.25], [0.15,0.25,0.35,0.5,0.7], [2.5,4,6,7.5,10]]
        tabla_alta =[[50,100000], [2,3], [1,1.5], [0.75,1.15], [0.3,0.45], [0.15,0.22], [2.5,3.75]]
        
        #Se hace un dataframe para cada nivel de voltaje, las columnas son el nivel de Zrelativa y las filas son los armonicos        
        index = [0,*range(2,51,1),'DATD']
        
        limites_baja = np.zeros((len(index),len(tabla_baja[0])))
        limites_media = np.zeros((len(index),len(tabla_media[0])))
        limites_alta = np.zeros((len(index),len(tabla_alta[0])))
        
        for n in range(0,len(index)-1):
            if index[n] < 11:
                limites_baja[n,:] = tabla_baja[1]
                limites_media[n,:] = tabla_media[1]
                limites_alta[n,:] = tabla_alta[1]
            elif index[n] < 17:
                limites_baja[n,:] = tabla_baja[2]
                limites_media[n,:] = tabla_media[2]
                limites_alta[n,:] = tabla_alta[2]
            elif index[n] < 23:
                limites_baja[n,:] = tabla_baja[3]
                limites_media[n,:] = tabla_media[3]
                limites_alta[n,:] = tabla_alta[3]
            elif index[n] < 35:
                limites_baja[n,:] = tabla_baja[4]
                limites_media[n,:] = tabla_media[4]
                limites_alta[n,:] = tabla_alta[4]
            else:
                limites_baja[n,:] = tabla_baja[5]
                limites_media[n,:] = tabla_media[5]
                limites_alta[n,:] = tabla_alta[5]
            if index[n] %2 ==0:
                limites_baja[n,:] = limites_baja[n,:]/4
                limites_media[n,:] = limites_media[n,:]/4
                limites_alta[n,:] = limites_alta[n,:]/4
        limites_baja[-1,:] = tabla_baja[6]
        limites_media[-1,:] = tabla_media[6]
        limites_alta[-1,:] = tabla_alta[6]
        
        #Aqui se crean los dataframes
        Ilim_baja = pd.DataFrame(columns = tabla_baja[0], index = index, data = limites_baja)
        Ilim_media = pd.DataFrame(columns = tabla_media[0], index = index, data = limites_media)
        Ilim_alta = pd.DataFrame(columns = tabla_alta[0], index = index, data = limites_alta)
        self.index = Ilim_baja.index
        
        #Se seleccionan los limites a devolver dependiendo del nivel de voltaje y Zrel
        
        if Vplanta < 69:
            df = Ilim_baja
        elif Vplanta < 161:
            df = Ilim_media
        else:
            df = Ilim_alta
                    
        for Imp in df.columns:
                if Zrel < Imp:
                    break
        
        self.Ilim = df[Imp].values


           
        # # Caso de hornos electricos
        # if cb_horno:
        #     self.Harm_lista = [Harm * 1.5 for Harm in self.Harm_lista]
        #     self.DATD = self.DATD * 1.5
        #     print("Limites de armonicos aumentaron por tratarse de horno.")
            
            
        #Vlim
        if Vplanta <= 1:
             Vlim = 6
        elif Vplanta <=35:
             Vlim = 5
        else:
             Vlim = 2
             
        self.Vlim = np.ones(np.shape(self.Ilim))*Vlim
        n = 0
        for x in index[0:-1]:
            if x % 2 == 0:
                self.Vlim[n] = self.Vlim[n]/4
            n = n+1
        self.Vlim = self.Vlim * LIM_VH
            
                
            
        
             
  
            
        
