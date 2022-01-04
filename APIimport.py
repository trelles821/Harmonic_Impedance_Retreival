# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:25:40 2021

@author: dtrelles
"""
from data.services.request_get_components import Components
from data.services.request_investigation import Investigation
from LimitesArmonicos import Limite
import pandas as pd
import numpy as np


class data_import(Investigation, Components):
    def __init__(self):
        
        
        Investigation.__init__(self)
        Components.__init__(self)
        # diccionarios y utilidades
        self.fases_VI_dic = { 'a': ['V1N','I1'],
                     'b': ['V2N','I2'],
                     'c': ['V3N','I3'],
                     'prom': ['V1N','V2N','V3N','I1','I2','I3'],
            }
        self.fases_P_dic = { 'a': 'P1',
                     'b': 'P2',
                     'c': 'P3',
                     'prom': 'P123',
            }
        
        #datos de ubicacion (Con estos la clase sabe donde pedir los datos que solicitaste (componente, tiempo y fase))
        self.componente = ''
        self.start_time = ''
        self.end_time = ''
        self.fase = ''     #a, b, c o prom. los demas codigos ya te arreglan todo para que solo se te devuelvan los datos de la fase que necesitas 
        
        #mediciones. el index es el timestamp. para los armonicos, las columnas son el numero
        self.Iharm = pd.DataFrame()
        self.Vharm = pd.DataFrame()
        self.Power = pd.DataFrame() #las columnas son P,Q,S, y FP
        self.IFund = pd.Series()
        self.VFund = pd.Series()
        
        #resumen de mediciones (Son los datos que se calculan a partir de las mediciones, ej: percentiles)
        self.Idemanda = 0
        self.Vnominal = 0
        self.Zrel = 0
        self.Vpercentil = 0
        self.Ipercentil = 0
        self.HarmElevados = 0 # Estos son los que tienen ya sea Ih o Vh elevados; NO son los que te pide el ECR (esos solo deben de tener Ih elevado)
         
        #datos de placa (todo lo que tiene que introducir el usuario)
        self.Vplanta = 0 # No necesariamente es el voltaje de base (si hay un transformador entre acometida y donde se midio)
        self.Zcc = 0
        self.cb_horno = False
    
    def get_datos_de_placa(self,Vplanta, Zcc, cb_horno):
        #Esta funcion ahorita parece una mamada pq solo estoy pasando los datos del main a la clase, pero supongo que luego que se implemente y obtenga los datos se obtengan de otro lugar (o el mismo codigo los obtenga dependiendo del nodo) ba a tener mas sentido
        
        self.Vplanta = Vplanta
        self.Zcc = Zcc
        self.cb_horno = cb_horno
        
    def locate_data(self,componente,start_time, end_time, fase):
        # Inicia sesion y obtiene los componentes
        # Si el API no esta en el mismo directorio tienes que cambiarlo con lo siguiente:
            #main_directory = getcwd()
            #chdir(r"C:\Users\dtrelles\Documents\GitHub\Sapphire_HTTP\code")
        #Al final de todo el codigo tienes que cerrar la sesion y volver al directorio original con:
            #self._Request_Sapphire__close_session() 
            #chdir(main_directory)
            
        self.componente = componente
        self.start_time = start_time
        self.end_time = end_time
        self.fase = fase
        
        
        
        self._Request_Sapphire__open_session()
        self.components = self.get(token = self.token)
        
    def get_harm(self, freq = '10MIN', harm = range(2,51), base = '10MIN'):
        #Toma los armonicos de voltaje y corriente
        df = pd.DataFrame()    
        
        fases_API = self.fases_VI_dic[self.fase]
        groups = []
        for h in harm:
            groups.append('HRMS_'+str(h))
        
        for f in fases_API:
            df_temporal, tree = self.get_data(self.componente, self.start_time, self.end_time, groups, f,  calculation_sync=freq, calculation_base=base,  token = self.token, components_dic = self.components)
            for titulo in df_temporal.columns:
                df_temporal.rename(columns={titulo: f + '_' + titulo}, inplace = True)
            df[df_temporal.columns]= df_temporal    
        
        #Te separa los datos segun su fase
        Va = df[df.columns[pd.Series(df.columns).str.startswith('V1N')]]
        Vb = df[df.columns[pd.Series(df.columns).str.startswith('V2N')]]
        Vc = df[df.columns[pd.Series(df.columns).str.startswith('V3N')]]
        Ia = df[df.columns[pd.Series(df.columns).str.startswith('I1')]]
        Ib = df[df.columns[pd.Series(df.columns).str.startswith('I2')]]
        Ic = df[df.columns[pd.Series(df.columns).str.startswith('I3')]]
        
        #Les cambia a las columnas para que sea el puro armonico (como int)
        Va.columns = Va.columns.str.replace('V1N_HRMS_','')
        Vb.columns = Vb.columns.str.replace('V2N_HRMS_','')
        Vc.columns = Vc.columns.str.replace('V3N_HRMS_','')
        Ia.columns = Ia.columns.str.replace('I1_HRMS_','')
        Ib.columns = Ib.columns.str.replace('I2_HRMS_','')
        Ic.columns = Ic.columns.str.replace('I3_HRMS_','')
        
        #Te devuelve el parametro que pediste
        if self.fase == 'a':
            V = Va; I = Ia
        elif self.fase == 'b':
            V = Vb; I = Ib
        elif self.fase == 'c':
            V = Vc; I = Ic
        elif self.fase == 'prom':
            V = (Va+Vb+Vc)/3; I = (Ia+Ib+Ic)/3
        
        # cambia los nombres de las columnas de strings a int (pa no batallar luego)
        V.columns = V.columns.astype(int)
        I.columns = V.columns.astype(int)
            
        #Guarda Vharm, Iharm en el objeto
        if freq == '10MIN':
            self.Vharm = V
            self.Iharm = I
        else:
            return V,I
        
    def get_power(self, freq = '10MIN', base = 'CYC'):
        # Agrega un df con P,Q,S y PF Fundamentales
        df = pd.DataFrame()    
        
        fases_API = self.fases_P_dic[self.fase]
        groups = ['ACTPWRF', 'REAPWRF', 'APPPWRF', 'PFF']
        
        df, tree = self.get_data(self.componente, self.start_time, self.end_time, groups, fases_API,  calculation_sync=freq, calculation_base=base,  token = self.token, components_dic = self.components)
        
        df.columns = ['P','Q','S','FP']
        self.Power = df
           
    def get_RMS(self, freq = '10MIN', base = '10MIN'):
        # Devuelve el RMS FUND de V y I para la fase seleccionada; tambien estima el voltaje base y corriente de demanda
        df = pd.DataFrame()    
        
        fases_API = self.fases_VI_dic[self.fase]
        groups = ['RMSFUND']
        
        for f in fases_API:
            df_temporal, tree = self.get_data(self.componente, self.start_time, self.end_time, groups, f,  calculation_sync=freq, calculation_base=base,  token = self.token, components_dic = self.components)
            for titulo in df_temporal.columns:
                df_temporal.rename(columns={titulo: f + '_' + titulo}, inplace = True)
            df[df_temporal.columns]= df_temporal
        
        #Te devuelve el parametro que pediste (como pd.Series)
        if self.fase == 'prom':
            V = df[df.columns[df.columns.str.startswith('V')]].mean(axis = 1) #Se ve muy confuso, pero ps es promediar las columnas que empiezan con V
            I = df[df.columns[df.columns.str.startswith('I')]].mean(axis = 1)
        else:
            V = df[fases_API[0] + '_' + groups[0]]
            I = df[fases_API[1] + '_' + groups[0]]
        
        
        #Los guarda en el objeto, y calcula voltaje base y Idemanda
        self.VFund = V
        self.Ifund = I
        self.Vnominal = np.mean(V)
        self.Idemanda = np.max(I)
        self.Zrel = (self.Vplanta/self.Zcc) * self.Idemanda
    
    def get_harmonic_angles(self, start_time = None, end_time = None, freq = '200MS', base = '200MS', harm = None):
        # Funcion para importar el angulo de cierto armonico, si le pides al sapphire una frecuencia mayor a 200MS, te va a devolver error. Hay unas tracalas para estimar el angulo 10 minutal, pero por lo pronto no las pongo;
        # Importar toda la semana a 200MS esta muy pesado, por lo que a esta funcion le pongo su propio start time y end time, para que solo se agarre un cachito
        
        df = pd.DataFrame()
        
        if harm is None:
            harm = self.HarmElevados
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
            
        if type(harm) == int or type(harm) == str:
            groups = 'IHRMSPA_'+str(harm*12)
        else:
            groups = []
            for h in harm:
                groups.append('IHRMSPA_'+str(h*12))
        
        fases_API = self.fases_VI_dic[self.fase]
        if fases_API == ['V1N','V2N','V3N','I1','I2','I3']:
            fases_API = ['V1N','I1']
           
            
        for f in fases_API:
            df_temporal, tree = self.get_data(self.componente, start_time, end_time, groups, f,  calculation_sync=freq, calculation_base=base,  token = self.token, components_dic = self.components)
            for titulo in df_temporal.columns:
                df_temporal.rename(columns={titulo: f + '_' + titulo}, inplace = True)
            df[df_temporal.columns]= df_temporal
        
        theta = df[df.columns[pd.Series(df.columns).str.startswith('V')]].replace(np.NaN, 0)
        phi = df[df.columns[pd.Series(df.columns).str.startswith('I')]].replace(np.NaN,0)
        
        return theta,phi #No los guarda en la clase pq no necesariamente son todo el timestamp
    
    
    
    def eval_harm(self):
        #Calcula los percentiles para todos los armonicos y evalua contra limites de corriente y voltaje

        #Checa que esten todos los armonicos [0,2:51], si no hay uno lo rellena con 0, para que no se deface, luego los acomoda 
        for h in [0,*range(2,51,1)]:
            if h not in self.Iharm.columns:
                self.Iharm[h] = 0
            if h not in self.Vharm.columns:
                self.Vharm[h] = 0
        self.Iharm = self.Iharm.reindex(sorted(self.Iharm.columns), axis=1)
        self.Vharm = self.Vharm.reindex(sorted(self.Vharm.columns), axis=1)
    
        #Se devuelven los 95 percentiles como porcentaje del valor base, en un array de [0,2:51]
        self.Vpercentil = self.Vharm.quantile(q=0.95).values*100/self.Vnominal
        self.Ipercentil = self.Iharm.quantile(q=0.95).values*100/self.Idemanda
        
        #Hace un array con los limites de corriente y voltaje, dependiendo de los parametros de la planta
        lim = Limite()
        lim.deterlim(self.Vplanta, self.Zrel, self.cb_horno, 0.3)
        
        #Te devuelve un array con los armonicos a analizar (los que tienen corriente O voltaje elevados)
        Iexcedente = (self.Ipercentil>lim.Ilim[0:len(self.Ipercentil)])
        Vexcedente = (self.Vpercentil>lim.Vlim[0:len(self.Vpercentil)])
        excedentes = Iexcedente*Vexcedente         #Tienen que salir los 2 por encima del limite
        self.HarmElevados = lim.index[np.where(excedentes)[0]]
        print('Harmonicos a analizar: {}'.format(self.HarmElevados.tolist()))
        
        
        
        