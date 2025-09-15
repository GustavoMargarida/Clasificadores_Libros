import cv2
import numpy as np
from pathlib import Path
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier

# Configuración
categorias = ['Comics', 'Libros', 'Manuscrito', 'Mecanografiado', 'Tickets']

# Datos C1 C2
tamaño = (306, 408)
train_path_C1C2 = Path('Muestra/Aprendizaje')
test_path_C1C2 = Path('Muestra/Test')

# Datos C3 C4

train_path_C3_C4 = Path('Muestra/AprendizajeTransformado')
test_path_C3_C4 = Path('Muestra/TestTransformado')

# Rutas modelos
modelo_path_C1 = Path('modelo_C1.pkl')
modelo_path_C2 = Path('modelo_C2.pkl')
modelo_path_C3 = Path('modelo_C3.pkl')
modelo_path_C4 = Path('modelo_C4.pkl')

seed = 34
entrenar = True  


def cargar_imagenes_y_etiquetas_C1_C2(base_path, categorias, tamaño):
    X, y, rutas = [], [], []
    for idx, categoria in enumerate(categorias):
        carpeta = base_path / categoria
        if not carpeta.exists():
            print(f"Carpeta no encontrada: {carpeta}")
            continue
        for imagen_path in carpeta.iterdir():
            if imagen_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(imagen_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"No se pudo leer: {imagen_path}")
                    continue
                img_redim = cv2.resize(img, tamaño, interpolation=cv2.INTER_LINEAR)
                X.append(img_redim.flatten())
                y.append(idx)
                rutas.append(imagen_path)
    return np.array(X), np.array(y), rutas


def cargar_imagenes_y_etiquetas_con_ruta_C3_C4(base_path, categorias, tamaño):
    X, y, rutas = [], [], []
    for idx, categoria in enumerate(categorias):
        carpeta = base_path / categoria
        if not carpeta.exists():
            print(f"Carpeta no encontrada: {carpeta}")
            continue
        for imagen_path in carpeta.iterdir():
            if imagen_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"No se pudo leer: {imagen_path}")
                    continue
                img_redim = cv2.resize(img, tamaño)
                X.append(img_redim.flatten())
                y.append(idx)
                rutas.append(imagen_path)  
    return np.array(X), np.array(y), rutas

# Cargar datos C1 C2
X_train_C1_C2, y_train_C1_C2, _ = cargar_imagenes_y_etiquetas_C1_C2(train_path_C1C2, categorias, tamaño)
X_test_C1_C2, y_test_C1_C2, _ = cargar_imagenes_y_etiquetas_C1_C2(test_path_C1C2, categorias, tamaño)
X_train_C1_C2 = X_train_C1_C2.astype(np.float32) / 255.0
X_test_C1_C2 = X_test_C1_C2.astype(np.float32) / 255.0

# Cargar datos C3 C4
X_test_C3_C4, y_test_C3_C4, rutas_test_C3_C4 = cargar_imagenes_y_etiquetas_con_ruta_C3_C4(test_path_C3_C4, categorias, tamaño)
X_train_C3_C4, y_train_C3_C4, rutas_train_C3_C4 = cargar_imagenes_y_etiquetas_con_ruta_C3_C4(train_path_C3_C4, categorias, tamaño)


X_train_C3_C4 = X_train_C3_C4.astype(np.float32) / 255.0
X_test_C3_C4 = X_test_C3_C4.astype(np.float32) / 255.0

# CLASIFICADOR 1
if entrenar or not modelo_path_C1.exists():
    print("Entrenando Clasificador C1...")
    clf1 = SVC(kernel='linear', C=1.0, random_state=seed)
    clf1.fit(X_train_C1_C2, y_train_C1_C2)
    joblib.dump(clf1, modelo_path_C1)
    print(f"Modelo C1 guardado en {modelo_path_C1}")
else:
    print(f"Cargando modelo C1 desde {modelo_path_C1}...")
    clf1 = joblib.load(modelo_path_C1)


y_pred_c1 = clf1.predict(X_test_C1_C2)
print("\nAccuracy C1:", accuracy_score(y_test_C1_C2, y_pred_c1))


# CLASIFICADOR 2
if entrenar or not modelo_path_C2.exists():
    print("Entrenando Clasificador C2 con PCA + LDA...")
    pca_c2 = PCA(n_components=30, random_state=seed)
    X_train_pca_C2 = pca_c2.fit_transform(X_train_C1_C2)
    X_test_pca_C2 = pca_c2.transform(X_test_C1_C2)

    lda = LDA(n_components=4)
    X_train_lda_C2 = lda.fit_transform(X_train_pca_C2, y_train_C1_C2)
    X_test_lda_C2 = lda.transform(X_test_pca_C2)

    clf2 = SVC(kernel='linear')
    clf2.fit(X_train_lda_C2, y_train_C1_C2)

    joblib.dump(clf2, modelo_path_C2)
    joblib.dump(pca_c2, 'pca_C2.pkl')
    joblib.dump(lda, 'lda_C2.pkl')

    print(f"Modelo C2 guardado.")
else:
    print("Cargando modelo C2...")
    clf2 = joblib.load(modelo_path_C2)
    pca_c2 = PCA(n_components=30, random_state=seed)
    X_train_pca_C2 = pca_c2.fit_transform(X_train_C1_C2)
    X_test_pca_C2 = pca_c2.transform(X_test_C1_C2)
    lda_C2 = LDA(n_components=4)
    X_train_lda_C2 = lda_C2.fit_transform(X_train_pca_C2, y_train_C1_C2)
    X_test_lda_C2 = lda_C2.transform(X_test_pca_C2)


y_pred_c2 = clf2.predict(X_test_lda_C2)
print("\nAccuracy C2:", accuracy_score(y_test_C1_C2, y_pred_c2))

# CLASIFICADOR 3
if entrenar or not modelo_path_C3.exists():
    print("Entrenando Clasificador C3 con PCA + SVM...")
    pca_c3 = PCA(n_components=30, random_state=seed)
    X_train_C3_pca = pca_c3.fit_transform(X_train_C3_C4)
    X_test_C3_pca = pca_c3.transform(X_test_C3_C4)

    clf3 = SVC(kernel='linear')
    clf3.fit(X_train_C3_pca, y_train_C3_C4)

    joblib.dump(clf3, modelo_path_C3)
    joblib.dump(pca_c3, 'pca_C3.pkl')

    print(f"Modelo C3 guardado.")
else:
    print("Cargando modelo C3...")
    clf3 = joblib.load(modelo_path_C3)
    pca_c3 = PCA(n_components=30, random_state=seed)
    X_train_C3_pca = pca_c3.fit_transform(X_train_C3_C4)
    X_test_C3_pca = pca_c3.transform(X_test_C3_C4)


y_pred_c3 = clf3.predict(X_test_C3_pca)
print("\nAccuracy C3:", accuracy_score(y_test_C3_C4, y_pred_c3))

# CLASIFICADOR 4

if entrenar or not modelo_path_C4.exists():
    print("Entrenando Clasificador C4 con PCA + LDA...")
    pca4 = PCA(n_components=30, random_state=seed)
    X_train_pca4 = pca4.fit_transform(X_train_C3_C4)
    X_test_pca4 = pca4.transform(X_test_C3_C4)
    lda4 = LDA(n_components=4)
    X_train_lda4 = lda4.fit_transform(X_train_pca4, y_train_C3_C4)
    X_test_lda4 = lda4.transform(X_test_pca4)
    
    clf4K = KNeighborsClassifier(n_neighbors=7)  
    clf4K.fit(X_train_lda4, y_train_C3_C4)
    
    joblib.dump(clf4K, modelo_path_C4)
    joblib.dump(pca4, 'pca_C4.pkl')
    joblib.dump(lda4, 'lda_C4.pkl')

    print(f"Modelo C4 guardado.")
else:
    print("Cargando modelo C4...")
    clf3 = joblib.load(modelo_path_C4)
    pca4 = PCA(n_components=30, random_state=seed)
    X_train_pca4 = pca4.fit_transform(X_train_C3_C4)
    X_test_pca4 = pca4.transform(X_test_C3_C4)
    lda4 = LDA(n_components=4)
    X_train_lda4 = lda4.fit_transform(X_train_pca4, y_train_C3_C4)
    X_test_lda4 = lda4.transform(X_test_pca4)


y_pred_KNN4 = clf4K.predict(X_test_lda4)


print("Accuracy:", accuracy_score(y_test_C3_C4, y_pred_KNN4))
