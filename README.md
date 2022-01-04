# Harmonic_Impedance_Retreival
Tool to estimate the harmonic impedances of an electrical system connected to the grid according to methods described in various research papers (example: https://ieeexplore.ieee.org/document/9640840/)

# BUSQA
Desarrollo de algoritmos en Python para la detección de contribución armónica externa.

Los algoritmos desarrollados en este Repositorio se integrarán a CloudPQ, donde Extrairán y analizarán mediciones PQZ para la detección de armónicos externos.

Los armónicos se analizan a través de las siguientes capas:

1. Verificación de armónicos de corriente excedentes.
2. Verificación de armónicos de voltaje elevados.
3. Separación de Modos a través de DBSCAN
4. Regresión Lineal Simple.
5. Regresión Lineal con Data Selection (R2).
6. Regresión Lineal con Data Selection (var).
8. ICA.
9. Ensamble de Métodos.
