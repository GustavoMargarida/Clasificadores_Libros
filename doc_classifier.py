import sys
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from rectificarHojas import extraer_descriptores

categorias = ['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets']
tamaño = (306, 408)


# Cargar clasificadores
clf1 = joblib.load('modelo_C1.pkl')
clf2 = joblib.load('modelo_C2.pkl')
clf3 = joblib.load('modelo_C3.pkl')
clf4 = joblib.load('modelo_C4.pkl')

# Cargar transformaciones
pca_c2 = joblib.load('pca_C2.pkl')
lda_c2 = joblib.load('lda_C2.pkl')
pca_c3 = joblib.load('pca_C3.pkl')
pca_c4 = joblib.load('pca_C4.pkl')
lda_c4 = joblib.load('lda_C4.pkl')

# Imagen a predecir
imagen_path = sys.argv[1]
img_color = cv2.imread(imagen_path, cv2.IMREAD_COLOR)
if img_color is None:
    print("No se pudo leer la imagen.")
    sys.exit(1)

# CLASIFICADOR 1
img_c1 = cv2.resize(img_color, tamaño)
x_c1 = img_c1.flatten().astype(np.float32) / 255.0
pred_c1 = clf1.predict([x_c1])[0]

# CLASIFICADOR 2
x_c2_pca = pca_c2.transform([x_c1])
x_c2_lda = lda_c2.transform(x_c2_pca)
pred_c2 = clf2.predict(x_c2_lda)[0]

# CLASIFICADOR 3 
if "ticket" not in imagen_path.lower():
    foto_rectificada = extraer_descriptores(imagen_path)
    img_gray = cv2.cvtColor(foto_rectificada, cv2.COLOR_BGR2GRAY)
else:
    img_gray = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

img_c34 = cv2.resize(img_gray, tamaño)
x_c34 = img_c34.flatten().astype(np.float32) / 255.0

x_c3_pca = pca_c3.transform([x_c34])
pred_c3 = clf3.predict(x_c3_pca)[0]

# CLASIFICADOR 4
x_c4_pca = pca_c4.transform([x_c34])
x_c4_lda = lda_c4.transform(x_c4_pca)
pred_c4 = clf4.predict(x_c4_lda)[0]

# RESULTADOS
print(f"Predicción C1: {categorias[pred_c1]}")
print(f"Predicción C2: {categorias[pred_c2]}")
print(f"Predicción C3: {categorias[pred_c3]}")
print(f"Predicción C4: {categorias[pred_c4]}")


plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title(f"C1: {categorias[pred_c1]} | C2: {categorias[pred_c2]} | C3: {categorias[pred_c3]} | C4: {categorias[pred_c4]}")
plt.axis('off')
plt.show()
