
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys



def cargar_puntos_desde_json(filename):
    f = open(filename)
    string_json=f.read()
    data = json.loads(string_json)
    
    puntos = {}
    
    for i in data['_via_img_metadata']:
        imagenes= data["_via_img_metadata"][i]["filename"]
        lista = []
        for j in data["_via_img_metadata"][i]["regions"]:
            x= j['shape_attributes']['cx']
            y= j['shape_attributes']['cy']
            lista.append((x,y))
        puntos[imagenes]=lista
    return puntos    
    

def umbralizar_gauss(imagen):
    sigma = 2 
    suavizada_x = gaussian_filter1d(imagen, sigma, axis=1)
    suavizada = gaussian_filter1d(suavizada_x, sigma, axis=0)
    filtered_image = cv2.GaussianBlur(suavizada, (5, 5), 0) 
    _, binaria = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    erosionada= erosionar(binaria,kernel)
    dilatada=dilatar(erosionada,kernel)
    return dilatada

def dilatar(imagen, kernel):
    op_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    opened_image = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, op_kernel)
    dilated_image = cv2.dilate(opened_image, kernel)
    return dilated_image

def erosionar(imagen, kernel):
    erosionada = cv2.erode(imagen, kernel)
    
    return erosionada


def ptos_interes_harris(imagen): 
    imgR = cv2.resize(imagen,(imagen.shape[1]//10,imagen.shape[0]//10))
    blockSize = 10
    ksize = 3
    k = 0.1
    esquinosidad = cv2.cornerHarris(imgR,blockSize,ksize,k)
    if len(imgR.shape) == 2:
        imgO = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
        
    return esquinosidad, imgO, imgR
   
def extraer_descriptores(direccionImagen):
    imagenOriginal = cv2.imread(direccionImagen, cv2.IMREAD_COLOR)
    
    if imagenOriginal is None:
        print(f"Error: No se pudo cargar la imagen {direccionImagen}.")
        return
    
    gray = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
    gauss_dilatado = umbralizar_gauss(gray)

    esquinosidad, imgInteres, imgR = ptos_interes_harris(gauss_dilatado)
    
    umbral = 0.1 * esquinosidad.max()
    imgInteres[esquinosidad > umbral] = [0, 0, 255]
    puntos_interes = cv2.ORB_create(edgeThreshold=13)
    keypoints, descriptores = puntos_interes.detectAndCompute(imgInteres, None)

    img_keypoints = cv2.drawKeypoints(imgInteres, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    des_test = descriptores
    kp_test = keypoints

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    '''descriptores_por_imagen = brief()  # Descriptores de las imágenes de referencia
    descriptores_referencia = np.vstack(list(descriptores_por_imagen.values()))'''
    matches = bf.match(des_test, des_test) #ref test
    matches = sorted(matches, key=lambda x: x.distance)

    quadrants = dividir_imagen_en_cuadrantes(imgInteres)

    
    best_points = mejores_esquinas_cuadrante(kp_test, matches, quadrants)
    
   
    for quadrant in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
        if quadrant not in best_points:
            print(f"Advertencia: Falta el punto {quadrant} en la imagen {direccionImagen}. Asignando valor 0,0.")
            best_points[quadrant] = (0,0)  

    distCuad = np.array([[0, 0], [2480, 0], [2480, 3508], [0, 3508]], np.float32)

    # Factor de escala usado para redimensionar
    scale_x = gauss_dilatado.shape[1] / imgR.shape[1]
    scale_y = gauss_dilatado.shape[0] / imgR.shape[0]

    # Escalar los puntos detectados al tamaño original
    srcCua = np.array([
        (best_points['top_left'][0] * scale_x, best_points['top_left'][1] * scale_y),
        (best_points['top_right'][0] * scale_x, best_points['top_right'][1] * scale_y),
        (best_points['bottom_right'][0] * scale_x, best_points['bottom_right'][1] * scale_y),
        (best_points['bottom_left'][0] * scale_x, best_points['bottom_left'][1] * scale_y)
    ], dtype=np.float32)

    try:
        # Intentar realizar la transformación de perspectiva
        persp_mat = cv2.getPerspectiveTransform(srcCua, distCuad)
        imgPerspectiva = cv2.warpPerspective(imagenOriginal, persp_mat, (2480, 3508))

        imagenRGB = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2RGB)
        imagenRGB2 = cv2.cvtColor(imgPerspectiva, cv2.COLOR_BGR2RGB)
        
       

    except cv2.error as e:
        print(f"Error al calcular la transformación de perspectiva para {direccionImagen}: {e}")
        # Si hay un error, puedes retornar o continuar con la siguiente imagen.
    return imagenRGB2

   
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def mejores_esquinas_cuadrante(kp_test, matches, quadrants):
    ep = 1e-6
    best_points = {}
    
    esquinas = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    for quadrant, (y1, y2, x1, x2) in quadrants.items():
        min_distance, max_priority = float('inf'), -float('inf')
        best_point = None
        
        corners = dict(zip(esquinas, [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]))
        corner = corners[quadrant]
        
        for match in matches:
            x, y = kp_test[match.trainIdx].pt
            if not (x1 <= x < x2 and y1 <= y < y2):
                continue
            
            dist_to_ref = match.distance
            dist_to_corner = euclidean_distance((x, y), corner)
            corner_priority = 1 / (dist_to_corner + ep)
            
            if dist_to_ref < min_distance:
                min_distance = dist_to_ref
                best_point = (x, y)
            
            if corner_priority > max_priority:
                max_priority = corner_priority
                best_point = (x, y)
        
        if best_point:
            best_points[quadrant] = best_point
    
    return best_points

def dividir_imagen_en_cuadrantes(imagen):
    h, w = imagen.shape[:2]
    center_x, center_y = h // 2, w // 2
    quadrants = {
        'top_left': (0, center_x, 0 ,center_y),
        'top_right': (0, center_x, center_y, w),
        'bottom_left': (center_x, h, 0, center_y),
        'bottom_right': (center_x, h, center_y, w)
    }
    return quadrants

