INSTRUCCIONES PARA EJECUTAR LA APLICACION DE CLASIFICACIÓN DE IMAGENES

1. Asegúrese de tener Python instalado.
   Para comprobarlo, abra la consola (cmd o PowerShell) y escriba:

       python --version

2. Navegue hasta la carpeta de ENTREGAFINALSK
   Por ejemplo:

       cd C:\Users\NombreDeUsuario\Documentos\ENTREGAFINALSK

4. Instale las dependencias necesarias
 
	python -m pip install joblib

	python -m pip install scikit-learn

   Si su sistema usa python3, puede que deba usar:

       python3 -m pip install joblib/scikit-learn

5. Ejecute el programa de entrenarModelos.py

       python entrenarModelos.py

6. La ejecución tardará un poco ya que entrena los modelos y los guarda en archivos para así agilizar después todas las pruebas y no necesitar de entrenar el modelo cada vez que se pruebe con una imagen nueva. En la consola saldrán los porcentajes de accuracy de cada clasificador y se crearán varios archivos.

7. Ejecutar el programa doc_classifier.py (ruta de imagen) (Muestra/Test/Comics/Comic_11.jpg)
	En muestra hay una carpeta de TestTransformado pero la original con las fotos con las que poder 	hacer pruebas es Test normal.

8. Verá el resultado de las predicciones realizadas por los diferentes clasificadores

NOTA: Dentro de Muestra hay un notebook con un intento temprano de haber realizado el ejercicio usando una red neuronal con pytorch.

