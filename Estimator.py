# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:47:37 2021

@author: dtrelles
"""

from APIimport import data_import
from LimitesArmonicos import Limite
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import Metodos


class separacion_de_contribucion(data_import):
    
    def __init__(self):
        
        data_import.__init__(self)
        
        
        
        
        
        #valores complejos
        self.phi_harm = pd.DataFrame(columns = range(2,51))       #angulos armonicos; se usa phi para cualquier angulo relacionado a corriente
        self.theta_harm = pd.DataFrame(columns = range(2,51))     #angulos armonicos; se usa theta para cualquier angulo relacionado a voltaje
        self.IharmComplex = pd.DataFrame(columns = range(2,51), dtype = 'complex128')
        self.VharmComplex = pd.DataFrame(columns = range(2,51), dtype = 'complex128')
        
        #Valores de alta frecuencia (tambien son complejos lol)
        # self.Iharm1MIN = pd.DataFrame(dtype = 'complex128')
        # self.Vharm1MIN = pd.DataFrame(dtype = 'complex128')
        # self.Iharm3SEC = pd.DataFrame(dtype = 'complex128')
        # self.Vharm3SEC = pd.DataFrame(dtype = 'complex128')
        # self.Iharm200MS = pd.DataFrame(dtype = 'complex128')
        # self.Vharm200MS = pd.DataFrame(dtype = 'complex128')
        
        #Valores de salida (fuentes separadas e impedancias)
        self.Iu = pd.DataFrame(dtype = 'complex128')
        self.Ic = pd.DataFrame(dtype = 'complex128')
        self.Vu = pd.DataFrame(dtype = 'complex128')
        self.Vc = pd.DataFrame(dtype = 'complex128')
        self.Zu = pd.DataFrame(dtype = 'complex128') 
        self.Zc = pd.DataFrame(dtype = 'complex128')
        
        #Valores para Separacion de Modos
        self.mod = pd.DataFrame() #Aqui se guarda el modo vs el timeindex; sirve para plotear los datos de salida tomando en cuenta el modo
        
    
    def get_harmonic_modes(self, harm = None, eps = 0.3):
        #Metodo para separar en modos. Los modos son bastante subjetivos, dependen del objetivo para el que quieras separar modos.
        #En este caso el objetivo es que se tengan Zh consistentes en cada modo, y que se cambie de modo pocas veces en el dia.
        # Por el segundo motivo se pone Z(t-1) y Z(t+1); para que el algoritmo tome en cuenta el estado justo antes  y justo despues del instante a analizarse.
        
        #Me preocupa que cuando haya 0 es mod.index quede desfazado o con NaNs, pero no tengo datos para probarlo todavia
        
        if harm is None:
            harm = self.HarmElevados
        elif type(harm) == pd.Index:
            harm = harm.index
            
        self.Tmod = pd.DataFrame(columns = harm, index = range(11)) #indices de tiempo que se usaron para cada modo 
        
        for h in harm:
            I = self.Iharm[h]
            V = self.Vharm[h]
            idx = (I !=0) & (V!=0)    #Te quita los valores en los que alguno de los 2 es zero
        
            #parametros
            x1 = I[idx].to_numpy().reshape(-1,1)/max(I)         #Ih
            x2 = V[idx].to_numpy().reshape(-1,1)/max(V)         #Vh
            x3 = x2/x1; #x3 = x3/max(x3)                                         #Zh
            x4 = np.append(x3[1:,:],x3[0,:]).reshape(-1,1)      #Zh(t-1); Zh en la medicion previa
            x5 = np.append(x3[-1,:],x3[0:-1,:]).reshape(-1,1)   #Zh(t+1); Zh en la medicion posterior
            
            X = np.concatenate((x1,x2,x3, x4, x5), axis = 1)
            
            samp = int(len(X)/50)                               #Para que se considere un modo, deben de contener por lo menos el 2% de los datos
            c = DBSCAN(eps = eps, min_samples = samp).fit_predict(X)    #El eps es bastante subjetivo, depende mucho de los parametros de entrada
            self.mod[h] = pd.Series(index = idx[idx==True].index, data = c) 
            
            for x in range(0, max(self.mod[h])+1):
                self.Tmod[h][x] = (self.mod[h]==x)
                
    
    def get_approx_harmonic_angles(self, harm = None, percentil = 0.95):
        #Obtiene el angulo armonico para cada modo. Como es muy pesado obtener todo el tiempo, solo busca el valor del n percentil y ese se lo asigna a todo el modo como si fuera fijo.
        
        
        
        if harm is None:
            harm = self.HarmElevados
        
        for h in harm:
            theta_harm = pd.Series()
            phi_harm = pd.Series()
            for x in range(0, max(self.mod[h])+1):
                I = self.Iharm[h][self.mod[h]==x]
                V = self.Vharm[h][self.mod[h]==x]
                timestamp = I[I == I.quantile(percentil, interpolation='higher')].index[0] # Te regresa el timestamp del n percentil de Ih
    
                start_time = timestamp - pd.Timedelta('5 minutes')
                end_time = timestamp + pd.Timedelta('5 minutes')
                theta,phi = self.get_harmonic_angles(start_time = start_time, end_time = end_time, harm = h)
                theta_harm = pd.concat([theta_harm,pd.Series(index = V.index, data = theta.iloc[:,0].mean())])
                phi_harm = pd.concat([phi_harm,pd.Series(index = I.index, data = phi.iloc[:,0].mean())])
            
            #Esto es para rellenar los espacios en el index que estan en el Iharm pero todavia no existen en el phi (se llena con nans, luego se van a rellenar)
            theta_harm = pd.concat([theta_harm,pd.Series(index = self.Vharm[5][~self.Vharm[5].index.isin(theta_harm.index)].index)])
            phi_harm = pd.concat([phi_harm,pd.Series(index = self.Iharm[5][~self.Iharm[5].index.isin(phi_harm.index)].index)])
            self.theta_harm[h] = theta_harm.sort_index()
            self.phi_harm[h] = phi_harm.sort_index()
            self.VharmComplex[h] = self.Vharm[h].astype('complex128')*(np.e**(1j*self.theta_harm[h]))
            self.IharmComplex[h] = self.Iharm[h].astype('complex128')*(np.e**(1j*self.phi_harm[h]))
                        
    
    def Regresion_lineal_DS(self,harm = None):
    #Prueba con los 3 metodos de DataSelection y usa las proxys para determinar cual es la Z que se escogio

        
        if harm is None:
            harm = self.HarmElevados.values
        elif type(harm) == pd.Index:
            harm = harm.index
            
        LIM_SLR = 0.9  #Limite para saber si fue una buena regresion lineal (se compara contra R2)
        self.R2 = pd.DataFrame(columns = harm, index = range(11), data = np.zeros((11, len(harm)))) #Resultado de la regresion (Deberia ser la mas chila pero ahorita es la ultima que haga lol)
        self.Zestimada = pd.DataFrame(columns = harm, index = range(11), dtype = 'complex128') # El valor de la pendiente (todavia no sabes si es Zu o Zc)
        self.C = pd.DataFrame(columns = harm, index = range(11), dtype = 'complex128') #Constante de la regresion
        
        self.Iharm1MIN = pd.DataFrame(index = self.Resample_index(self.Iharm.index, '1MIN'), dtype = 'complex128')
        self.Vharm1MIN = pd.DataFrame(index = self.Iharm1MIN.index, dtype = 'complex128')
        self.Iharm3SEC = pd.DataFrame(index = self.Resample_index(self.Iharm.index, '3SEC'), dtype = 'complex128')
        self.Vharm3SEC = pd.DataFrame(index = self.Iharm3SEC.index, dtype = 'complex128')
        self.Iharm200MS = pd.DataFrame(index = self.Resample_index(self.Iharm.index, '200MS'), dtype = 'complex128')
        self.Vharm200MS = pd.DataFrame(index = self.Iharm200MS.index, dtype = 'complex128')
    
        for h in harm:
            for x in range(0, max(self.mod[h])+1):
                
                #Estos son pa quedarte con el valor de la mejor regresion
                R2max = 0
                m_max = 0 + 0j
                b_max = 0 + 0j
                idx_10min = (self.mod[h]==x).index
                # Se hace un Loop a diferentes frecuencias (10 min, 1min, 3s), para probar cada uno de los tres metodos
                for freq in ['10MIN', '1MIN', '3SEC']:
                    #Es para poder agarrar los armonicos a la frecuencia que se esta usando
                    if freq != '10MIN':
                        I_attr = 'Iharm'+freq
                        V_attr = 'Vharm'+freq
                        idx = self.Resample_index(self.mod[h][idx_10min].index, freq)
                    else:
                        I_attr = 'IharmComplex'
                        V_attr = 'VharmComplex'
                        idx = idx_10min
                        
                    
                    #Aqui se hace un if para ver si se van a importar nuevos datos, si se necesitan mas datos para 1 armonico, se importa pa todos los excedentes de un jalon
                    if getattr(self,I_attr).isnull().values.all() or getattr(self,V_attr).isnull().values.all():
                        self.get_high_frequency_harmonics(freq)
                    
                    #Toma los armonicos   (Esta medio mafufo esto pero es pq el la funcion de resamplear index agarra desde 5 minutos antes que empiece la medicion, entonces hay que cortarle ese pedacito)
                    I = getattr(self,I_attr)
                    V = getattr(self,V_attr)
                    idx = idx[(idx >= min(I.index)) * (idx <= max(I.index))]
                    I = I[h][idx]
                    V = V[h][idx]
                    
                    #Regresion Lineal Simple (RLS)
                    cal, m, b = Metodos.RegresionLinealSimple(I,V)
                    metodo = 'RLS'
                    if (cal >= R2max): 
                        R2max = cal;    m_max = m;  b_max = b;  metodo = 'RLS'    #Esto o puse en una linea pq lo uso 3 veces y esta muy simple, si la R2 es mayor que lo que se tenia antes, se remplazan los valores de R2, Z y C
                    if cal<LIM_SLR: #Regresion metodo de Varianza (RLV)
                        cal, m, b = Metodos.RegresionLinealVAR(I,V)
                        if (cal >= R2max): 
                            R2max = cal;    m_max = m;  b_max = b;  metodo = 'RLVAR'
                    if cal<LIM_SLR:     #Regresion Lineal R2 (RLR2)
                        cal, m, b = Metodos.RegresionLinealR2(I,V)
                        if cal >= R2max:
                            R2max = cal;    m_max = m;  b_max = b;  metodo = 'RLR2'
                    
                        
                    if cal>LIM_SLR:
                        print('arm: {0}; modo: {1}; R2: {2:.3f}; Z: {3:.3f}; C: {4:.3f}; freq: {5}, metodo: {6}'.format(h,x,cal,m,b,freq, metodo))
                        self.R2[h][x] = cal
                        self.Zestimada[h][x] = m
                        self.C[h][x] = b
                        break
                    
                    elif freq == '3SEC':
                        print('No se obtuvo una buena Regresión para el armonico {0}, modo {1} (R2 = {2:.3f}) (metodo: {3}).'.format(str(h),x,R2max, metodo))
                        self.R2[h][x] = R2max
                        self.Zestimada[h][x] = m_max
                        self.C[h][x] = b_max
            
    
    def Seleccion_de_impedancia_RL(self,harm = None):
        
        
        
        
        if harm is None:
            harm = self.HarmElevados.values
        elif type(harm) == pd.Index:
            harm = harm.index
        
        
        self.Zescogida_RL = pd.DataFrame(columns = harm, index = range(11)) # te dice si la pendiente que caluclaste es Zu o Zc        
        self.resp_LR = pd.DataFrame(columns = harm, index = range(11)) # Responsabilidad (Cliente o Utilidad segun Regresion Lineal)
        self.Zu_por_modo = pd.DataFrame(columns = harm, index = range(11)) # Zu por modo lol
        self.Zc_por_modo = pd.DataFrame(columns = harm, index = range(11)) # 
        for h in harm:
            for x in range(0, max(self.mod[h])+1):
            
                idx = self.mod[h]==x
                ZuProxy = self.Zcc*h*1j
                ZcProxy = (self.Vnominal**2)/self.Power['S'][idx]
                
                m = self.Zestimada[h][x]
                I = np.mean(self.Iharm[h][idx])
                b = self.C[h][x]
                
                #Verificar Calidad de Zu, Zc obtenidas
                if abs(ZuProxy)*10 < abs(np.mean(ZcProxy)):
                    ZcProxy = abs(ZuProxy)*10
                    
                Zcrit = (abs(ZuProxy)+abs(np.mean(ZcProxy)))/2
                if abs(m) < Zcrit:
                    self.Zescogida_RL[h][x] = 'Zu'
                    self.Zu_por_modo[h][x] = m
                    self.Zc_por_modo[h][x] = ZcProxy
                    if abs(m*I)>abs(b):
                        self.resp_LR[h][x] = 'c'
                    else:
                        self.resp_LR[h][x] = 'u'
                            
                else:
                    self.Zescogida_RL[h][x] = 'Zc'
                    self.Zc_por_modo[h][x] = m
                    self.Zu_por_modo[h][x] = ZuProxy
                    if abs(m*I)>abs(b):
                        self.resp_LR[h][x] = 'u'
                    else:
                        self.resp_LR[h][x] = 'c'
                                    
    
    def Seleccion_de_impedancia_CI(self,harm=None):
        # Esta función está diseñada para implementar el método de impedancia
        # crítica de Wilsun Xu, Chun Li y Tayjasanant T. Este método sirve para 
        # determinar el mayor responsable de la contaminación armónica en 
        # corriente en el punto de conexion.
        #
        # definiciones:
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
        
        if harm is None:
            harm = self.HarmElevados.values
        elif type(harm) == pd.Index:
            harm = harm.index
        
        
        self.IET = pd.DataFrame(columns = harm, index = range(11))  # Impedance Error Tolerance Index (Eq. 13 paper critical impedance)
        self.CI = pd.DataFrame(columns = harm, index = range(11))   # Critical Impedance (Eq. 12 paper critical impedance)
        self.resp_CI = pd.DataFrame(columns = harm, index = range(11)) #Responsabilidad (Cliente o Utilidad de acuerdo a metodo de CI)
        
        
        for h in harm:
            for x in range(0, max(self.mod[h])+1):
                Vpcc = np.mean(self.VharmComplex[h][self.Tmod[h][x]])
                Ipcc = np.mean(self.IharmComplex[h][self.Tmod[h][x]])
                Zu = self.Zu_por_modo[h][x]
                Zc = self.Zc_por_modo[h][x]
                
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
                self.CI[h][x] = 2*np.abs(Eu/I)*np.sin(theta+beta)  # Critical impedance
                
                # impedance error tolerance index (Eq. 13)
                self.IET[h][x] = np.abs((np.abs(self.CI[h][x])-np.abs(Z))/Z)
                
                # Si la CI > 0, la culpa es del cliente
                if self.CI[h][x] > 0:      
                    self.resp_CI[h][x] = 'c'
                else: # Si CI < 0
                    # Si la magnitud de la CI es mayor que la de la Zmax: culpa de la red
                    if np.abs(self.CI[h][x]) > np.abs(Zmax):
                        self.resp_CI[h][x] = 'u'
                    
                    # Si la magnitud de la CI es menor que la de la Zmin: culpa del cliente
                    elif np.abs(self.CI[h][x]) < np.abs(Zmin):
                        self.resp_CI[h][x] = 'c'
                        
                    # Si CI está fuera de rango, no es conclusivo el test
                    else:
                        self.resp_CI[h][x] = 'not conclusive'       
    
    
    def Comparacion_de_metodos_de_seleccion(self, harm = None):
        # Si resp_LR y resp_CI no son congruentes, las Z's se invierten
        
        if harm is None:
            harm = self.HarmElevados.values
        elif type(harm) == pd.Index:
            harm = harm.index
            
        self.Zescogida = pd.DataFrame(columns = harm, index = range(11)) # te dice si la pendiente que caluclaste es Zu o Zc        
        
        
        for h in harm:
            for x in range(0, max(self.mod[h])+1):
                if (self.resp_LR[h][x] != self.resp_CI[h][x]) and self.resp_CI[h][x] != 'not conclusive':
                    if self.Zescogida_RL[h][x] == 'Zu':
                        self.Zescogida[h][x] = 'Zc'
                        self.Zc_por_modo[h][x] = self.Zu_por_modo[h][x]
                        self.Zu_por_modo = self.Zcc * h * 1j
                    else:
                        self.Zescogida[h][x] = 'Zu'
                        self.Zu_por_modo[h][x] = self.Zc_por_modo[h][x]
                        self.Zc_por_modo[h][x] = (self.Vnominal**2)/self.Power['S'][self.Tmod[h][x]]       
                else:
                    self.Zescogida[h][x] = self.Zescogida_RL[h][x]
                        
    def Reconstruccion_de_fuentes(self, harm = None):
    
        if harm is None:
            harm = self.HarmElevados.values
        elif type(harm) == pd.Index:
            harm = harm.index
        
        #Valores de salida (fuentes separadas e impedancias)
        self.Iu = pd.DataFrame(columns = harm, index = self.Iharm.index, dtype = 'complex128')
        self.Ic = pd.DataFrame(columns = harm, index = self.Iu.index, dtype = 'complex128')
        self.Vu = pd.DataFrame(columns = harm, index = self.Iu.index, dtype = 'complex128')
        self.Vc = pd.DataFrame(columns = harm, index = self.Iu.index, dtype = 'complex128')
        self.Zu = pd.DataFrame(columns = harm, index = self.Iu.index, dtype = 'complex128') 
        self.Zc = pd.DataFrame(columns = harm, index = self.Iu.index, dtype = 'complex128')
            
        
        for h in harm: 
            for x in range(0, max(self.mod[h])+1):
                
                self.Zu[h][self.Tmod[h][x]] = self.Zu_por_modo[h][x]
                self.Zc[h][self.Tmod[h][x]] = self.Zc_por_modo[h][x]
        
        for h in harm:
            #Hay que rellenar los Nans que salieron por los espacios que tenian mod = -1; ffill te rellena el valor con el valor anterior, bfill con el proximo. Hay que usar los 2 por si el primer y ultimo valor son nans
            if self.Zu[h].isna().any():
                #Esta mamda la tuve que hacer porque fillna() no jalaba con complejos
                Zu_mag = abs(self.Zu[h])
                Zu_ang = pd.Series(index = self.Zu[h].index, data = np.angle(self.Zu[h]))
                Zu_mag.fillna(method = 'ffill', inplace = True)
                Zu_ang.fillna(method = 'ffill', inplace = True)
                self.Zu[h] = Zu_mag.astype('complex128')*(np.e**(1j*Zu_ang))
                
                Zc_mag = abs(self.Zc[h])
                Zc_ang = pd.Series(index = self.Zc[h].index, data = np.angle(self.Zc[h]))
                Zc_mag.fillna(method = 'ffill', inplace = True)
                Zc_ang.fillna(method = 'ffill', inplace = True)
                self.Zc[h] = Zc_mag.astype('complex128')*(np.e**(1j*Zc_ang))
                
                self.phi_harm[h].fillna(method = 'ffill', inplace = True)
                self.theta_harm[h].fillna(method = 'ffill', inplace = True)
            if self.Zu[h].isna().any():
                Zu_mag = abs(self.Zu[h])
                Zu_ang = pd.Series(index = self.Zu[h].index, data = np.angle(self.Zu[h]))
                Zu_mag.fillna(method = 'bfill', inplace = True)
                Zu_ang.fillna(method = 'bfill', inplace = True)
                self.Zu[h] = Zu_mag.astype('complex128')*(np.e**(1j*Zu_ang))
                
                Zc_mag = abs(self.Zc[h])
                Zc_ang = pd.Series(index = self.Zc[h].index, data = np.angle(self.Zc[h]))
                Zc_mag.fillna(method = 'bfill', inplace = True)
                Zc_ang.fillna(method = 'bfill', inplace = True)
                self.Zc[h] = Zc_mag.astype('complex128')*(np.e**(1j*Zc_ang))

                self.phi_harm[h].fillna(method = 'bfill', inplace = True)
                self.theta_harm[h].fillna(method = 'bfill', inplace = True)
            
            #Esto es para rellenar los valores de los complex para los que no se tenia angulo
            self.IharmComplex[h] = self.Iharm[h].astype('complex128')*(np.e**(1j*self.phi_harm[h]))
            self.VharmComplex[h] = self.Vharm[h].astype('complex128')*(np.e**(1j*self.theta_harm[h]))
            # Generar Fuentes Iu, Ic, Vu, Vc
            self.Iu[h] = (self.VharmComplex[h]/self.Zu[h]) + self.IharmComplex[h]
            self.Ic[h] = (self.VharmComplex[h]/self.Zc[h]) - self.IharmComplex[h]
            self.Vu[h] = self.Iu[h]*self.Zu[h]
            self.Vc[h] = self.Ic[h]*self.Zc[h]
        
    def Exportar_fuentes(self):
        
        # Devolver CSV con Iu, Ic, Vu, Vc, Zu, Zc para cada armónico
        writer = pd.ExcelWriter('Contribuciones.xlsx', engine='xlsxwriter')
        self.Vu.to_excel(writer, sheet_name='Vu')
        self.Vc.to_excel(writer, sheet_name='Vc')
        self.Iu.to_excel(writer, sheet_name='Iu')
        self.Ic.to_excel(writer, sheet_name='Ic')
        self.Zu.to_excel(writer, sheet_name='Zu')
        self.Zc.to_excel(writer, sheet_name='Zc')
        writer.save()
        
    
    
    
    def Estimate_filter_response(self):
        
        self.Ic_predicted = pd.DataFrame(columns = range(2,51), dtype = 'complex128')
        self.Iu_predicted = pd.DataFrame(columns = range(2,51), dtype = 'complex128')
        self.Zc_post_instalacion = pd.DataFrame(index = self.Iharm.index, columns = range(2,51), dtype = 'complex128')
        self.Ipcc_post_instalacion = pd.DataFrame(index = self.Iharm.index, columns = range(2,51), dtype = 'complex128')
        
        for h in [0,*range(2,51,1)]:
            self.Zc_post_instalacion[h] = 5j    #Aqui debe ser la impedancia del trafo principal pero todavia no la pongo
            if h not in self.Iu.columns:
                #Todo esto va a jalar mejor cuando se tenga un buen aproximado de Zc post instalacion
                self.Zu[h] = self.Zcc*h*1j
                self.Zc[h] = (self.Vnominal**2)/self.Power['S']
                self.Ic[h] = self.Iharm[h]*(self.Zu[h]+self.Zc[h])/self.Zc[h]
                self.Iu[h] = 0
            
            
            self.Ipcc_post_instalacion[h] = ((self.Ic[h]*self.Zc_post_instalacion[h])-(self.Iu[h]*self.Zu[h]))/(self.Zu[h]+self.Zc_post_instalacion[h])
                
        self.Zu = self.Zu.reindex(sorted(self.Zu.columns), axis=1)
        self.Zc = self.Zc.reindex(sorted(self.Zc.columns), axis=1)
        self.Iu = self.Iu.reindex(sorted(self.Iu.columns), axis=1)
        self.Ic = self.Ic.reindex(sorted(self.Ic.columns), axis=1)
        self.Ipcc_post_instalacion = self.Ipcc_post_instalacion.reindex(sorted(self.Ipcc_post_instalacion.columns), axis=1)
        
        #Se devuelven los 95 percentiles como porcentaje del valor base, en un array de [0,2:51]
        self.Ipcc_post_instalacion['DATD'] = np.sqrt((abs(self.Ipcc_post_instalacion)**2).sum(axis = 1))
        self.Ipercentil_post_instalacion = abs(self.Ipcc_post_instalacion).quantile(q=0.95).values*100/self.Idemanda
        
        #Hace un array con los limites de corriente y voltaje, dependiendo de los parametros de la planta
        lim = Limite()
        lim.deterlim(self.Vplanta, self.Zrel, self.cb_horno, 0.3)
        
        #Te devuelve un array con los armonicos a analizar (los que tienen corriente O voltaje elevados)
        Iexcedente = (self.Ipercentil_post_instalacion>lim.Ilim)
        
        if Iexcedente.any():
            print('No se recomienda una solución puramente pasiva para los armonicos: {}'.format(lim.index[Iexcedente].tolist()))
        else:
            print('Una solución puramente pasiva es posible.')
                
    
    #Utils
    
    def Resample_index(self,T,freq):
        if freq == '200MS':
            freq = '0.2S'
        elif freq == '3SEC':
            freq = '3S'
            
        idx = pd.DatetimeIndex(data = [])
        delta = pd.Timedelta('5 minutes')
        for dt in T:
            Ti = dt -delta
            Tf = dt + delta - pd.Timedelta(freq)
            Trange = pd.date_range(start = Ti, end = Tf, freq = freq)
            idx = idx.append(Trange)
            
        return idx        
    
            
    def get_high_frequency_harmonics(self,freq, harm = None):
        #Creo que no esta jalando, devuelve el mismo angulo para todos los modos
        
        #solo estoy tomando los datos a una frecuencia mas alta, aqui el pedo es saber cual angulo va con cual magnitud
        if harm is None:
            harm = self.HarmElevados
        
        if freq == '200MS':
            base = freq
        else:
            base = '3SEC'
            
            
        V,I = self.get_harm(freq = freq, harm = harm, base = base)
        V = V.astype('complex128')
        I = I.astype('complex128')
        
        for h in harm:
            theta_hf = pd.Series()
            phi_hf = pd.Series()
            
            for x in range(-1, max(self.mod[h])+1):
                idx = self.Resample_index(self.mod[self.mod[h]==x].index,freq)
                theta = np.angle(self.VharmComplex[h][self.mod[h]==x][0])
                phi = np.angle(self.IharmComplex[h][self.mod[h]==x][0])
                theta_hf = pd.concat([theta_hf,pd.Series(index = idx, data = theta)])
                phi_hf = pd.concat([phi_hf,pd.Series(index = idx, data = phi)])
                
        
            I[h] = I[h]*(np.e**(1j*phi_hf[h]))
            V[h] = V[h]*(np.e**(1j*theta_hf[h]))
        
        if freq == '1MIN':
            self.Iharm1MIN = I
            self.Vharm1MIN = V
        elif freq == '3SEC':
            self.Iharm3SEC = I
            self.Vharm3SEC = V
        elif freq == '200MS':
            self.Iharm200MS = I
            self.Vharm200MS = V
            
            
    
            
            
            
            
            
            
            
            
            
            
