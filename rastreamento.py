import cv2 as cv
import numpy as np

# Usados para armazenar as coordenadas de cada carrinho
# para desenhar o rastro
listaGreen = []
listaYellow = []

cap = cv.VideoCapture('video4.MTS')
while(1):
    # Pegar cada frame
    _, frame = cap.read()
    if frame is None:
        break
    
    # Converter de RGB para HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Define os intervalos de cores HSV
    lower_orange = np.array([165,50,50])
    upper_orange = np.array([175,255,255])
    redLower = (-10, 50, 50)
    redUpper = (10, 255, 255)
    
    # Criando máscaras para pegar apenas cores vermelhas e laranjas na imagem
    maskg = cv.inRange(hsv, redLower, redUpper)
    mask = cv.inRange(hsv, lower_orange, upper_orange)
    # Utilização de operadores morfológicos para preencher os espaços não detectados 
    mask = cv.erode(mask, None, iterations=5)
    mask = cv.dilate(mask, None, iterations=20)
    maskg = cv.erode(maskg, None, iterations=4)
    maskg = cv.dilate(maskg, None, iterations=25)

    # Verifica se há contornos no frame para máscara da cor vermelha
    cntGreen = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    centerGreen = None
    # Verifica se há contornos no frame para máscara da cor vermelha
    cntYellow = cv.findContours(maskg.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    centerYellow = None
    
    # Se pelo menos um contorno verde foi encontrado
    if len(cntGreen) > 0:
        # Encontrar o maior contorno da mascara, em seguida, usar-lo para calcular o círculo de fecho mínimo e
        # centróide
        # Desenha um contorno verde em volta do carrinho vermelho
        cGreen = max(cntGreen, key=cv.contourArea)
        rectGreen = cv.minAreaRect(cGreen)
        boxGreen = cv.boxPoints(rectGreen)
        boxGreen = np.int0(boxGreen)
        MGreen = cv.moments(cGreen)
        centerGreen = (int(MGreen["m10"] / MGreen["m00"]), int(MGreen["m01"] / MGreen["m00"]))
        listaGreen.append(centerGreen)
        cv.drawContours(frame, [boxGreen], 0, (0, 255, 0), 2)
        
    # Se pelo menos um contorno amarelo foi encontrado
    if len(cntYellow) > 0:
        # Desenha um contorno amarelo em volta do carrinho laranja
        cYellow = max(cntYellow, key=cv.contourArea)
        rectYellow = cv.minAreaRect(cYellow)
        boxYellow = cv.boxPoints(rectYellow)
        boxYellow = np.int0(boxYellow)
        MYellow = cv.moments(cYellow)
        centerYellow = (int(MYellow["m10"] / MYellow["m00"]), int(MYellow["m01"] / MYellow["m00"]))
        listaYellow.append(centerYellow)
        cv.drawContours(frame, [boxYellow], 0, (0, 255, 255), 2)
        

    # Desenhar o rastro de cada carrinho encontrado
    for i in range(len(listaGreen)-1):
        cv.line(frame, listaGreen[i], listaGreen[i+1], (0, 0, 255), 5)
    
    for i in range(len(listaYellow)-1):
        cv.line(frame, listaYellow[i],  listaYellow[i+1], (255, 0, 0), 5)
        
    cv.imshow('frame',frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
