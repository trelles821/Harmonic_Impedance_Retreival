# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:55:35 2021

@author: dtrelles
"""

from os import chdir
from os import getcwd
from Estimator import separacion_de_contribucion

main_directory = getcwd()
chdir(r"C:\Users\dtrelles\Documents\GitHub\Sapphire_HTTP\code")

Vplanta = 123000
Zcc = 10
cb_horno = False

 
a = separacion_de_contribucion()
a.get_datos_de_placa(Vplanta, Zcc, cb_horno)
a.locate_data('SolaireDirect', '15/03/20 10:00:00', '22/03/20 10:00:00', 'prom')
#try:
a.get_harm()
a.get_power()
a.get_RMS()
a.eval_harm()


a.get_harmonic_modes()
a.get_approx_harmonic_angles()
a.Regresion_lineal_DS()     #Aqui dentro se importan mayores frecuencias si es necesario
a.Seleccion_de_impedancia_RL()
a.Seleccion_de_impedancia_CI()
a.Comparacion_de_metodos_de_seleccion()
a.Reconstruccion_de_fuentes()
a.Estimate_filter_response()

# except:
#     print('Error')

a._Request_Sapphire__close_session() 
chdir(main_directory)

