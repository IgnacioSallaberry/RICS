# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:06:58 2020

@author: Ignacio Sallabery
"""
import numpy
from PIL import Image
import time

####  ------------------------    PARA UNA IMAGEN    -------------------

im = Image.open('C:\\Users\\ETCasa\\Desktop\\photoncount_GFP_C001T001.tif')   #importa la imagen .tif
#im.show()     #muestra la imagen importada
#
imarray = numpy.array(im) #crea una matriz de valores de la imagen importada
####  imarray[FILA][COLUMNA]

pixel_en_el_tiempo=[]
f=0
while f<len(imarray):
    p=0
    while p<len(imarray):
        ###  imarray[FILA][COLUMNA]
        pixel_en_el_tiempo.append([imarray[f][p]]) #crea una matriz de valores de la imagen importada
        p+=1
    f+=1
####  ------------------------------------------------------------------


####  ------------------------------------------------------------------
##                  creo vector con el nombre de las imágenes
####  ------------------------------------------------------------------
i=2
indice=[]
while i<100:
    
    if i<10:
        indice.append(f'00{i}')
    else:
        indice.append(f'0{i}')
        
    i+=1
####  ------------------------------------------------------------------

            


####  ------------------------    PARA UN CONJUNTO DE IMAGENES    -------------------
start = time.time()
print("hello")

i=1
while i<len(indice)+1:
    im = Image.open(f'C:\\Users\\ETCasa\\Desktop\\darkcountsFV1000UNSAM\\photoncountCH1\\photoncount GFP_C001T{indice[i]}.tif')   #importa la imagen .tif
    imarray = numpy.array(im) #crea una matriz de valores de la imagen importada
    h=0
    f=0
    while f<len(imarray):
        p=0
        while p<len(imarray):
            pixel_en_el_tiempo[h].append(numpy.array(im)[f][p]) #crea una matriz de valores de la imagen importada
            h+=1

            p+=1
        f+=1
    print(i)
    i+=1
    
end = time.time()
print(end - start)
####  ------------------------------------------------------------------


start = time.time()
print("hello")
end = time.time()
print(end - start)
