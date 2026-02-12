"""
ETAPA 1: AUTOENCODER PARA FASHION MNIST
<<<<<<< HEAD

=======

>>>>>>> etapa2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("ETAPA 1: AUTOENCODER PARA FASHION MNIST")
print("="*60)

# ------------------------------------------------------------
# 1. CARGAR DATOS
# ------------------------------------------------------------
print("\n1. Cargando Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

<<<<<<< HEAD
# Aplanar las imágenes de 28x28 a 784 píxeles
=======
# Aplanar imágenes
>>>>>>> etapa2
X_train_flat = x_train.reshape(x_train.shape[0], -1)
X_test_flat = x_test.reshape(x_test.shape[0], -1)

# Crear DataFrames
data = pd.DataFrame(X_train_flat)
data_test = pd.DataFrame(X_test_flat)

<<<<<<< HEAD
# Insertar la columna de etiquetas al inicio
=======
# Insertar etiquetas
>>>>>>> etapa2
data.insert(0, 'label', y_train)
data_test.insert(0, 'label', y_test)

print(f" data shape: {data.shape}")
print(f" data_test shape: {data_test.shape}")

# ------------------------------------------------------------
<<<<<<< HEAD
# 2. SEPARAR CARACTERÍSTICAS (X) Y ETIQUETAS (y)
=======
# 2. SEPARAR X y y
>>>>>>> etapa2
# ------------------------------------------------------------
print("\n2. Separando características y etiquetas...")
X = data.iloc[:, 1:]
X_test = data_test.iloc[:, 1:]
y = data.iloc[:, 0]
y_test = data_test.iloc[:, 0]

print(f" X shape: {X.shape}")
print(f" X_test shape: {X_test.shape}")

# ------------------------------------------------------------
# 3. NORMALIZAR
# ------------------------------------------------------------
print("\n3. Normalizando datos...")
X = X.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print("✅ Datos normalizados")

# ------------------------------------------------------------
<<<<<<< HEAD
# 4. DEFINIR DIMENSIÓN DEL ESPACIO LATENTE
=======
# 4. DIMENSIÓN LATENTE
>>>>>>> etapa2
# ------------------------------------------------------------
latent_dim = 32
print(f"\n4. Dimensión del espacio latente: {latent_dim}")

# ------------------------------------------------------------
<<<<<<< HEAD
# 5. CONSTRUIR ENCODER
# ------------------------------------------------------------
print("\n5. Construyendo ENCODER...")

=======
# 5. ENCODER
# ------------------------------------------------------------
print("\n5. Construyendo ENCODER...")
>>>>>>> etapa2
encoder_input = keras.Input(shape=(784,), name='encoder_input')
x = layers.Dense(128, activation='relu', name='encoder_dense_1')(encoder_input)
x = layers.Dense(64, activation='relu', name='encoder_dense_2')(x)
latent = layers.Dense(latent_dim, activation='relu', name='espacio_latente')(x)
<<<<<<< HEAD

encoder = models.Model(encoder_input, latent, name='encoder')
encoder.summary()

# ------------------------------------------------------------
# 6. CONSTRUIR DECODER
# ------------------------------------------------------------
print("\n6. Construyendo DECODER...")

=======
encoder = models.Model(encoder_input, latent, name='encoder')
#encoder.summary()

# ------------------------------------------------------------
# 6. DECODER
# ------------------------------------------------------------
print("\n6. Construyendo DECODER...")
>>>>>>> etapa2
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(64, activation='relu', name='decoder_dense_1')(decoder_input)
x = layers.Dense(128, activation='relu', name='decoder_dense_2')(x)
decoder_output = layers.Dense(784, activation='sigmoid', name='decoder_output')(x)
<<<<<<< HEAD

decoder = models.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

# ------------------------------------------------------------
# 7. CONSTRUIR AUTOENCODER
# ------------------------------------------------------------
print("\n7. Construyendo AUTOENCODER...")

=======
decoder = models.Model(decoder_input, decoder_output, name='decoder')
#decoder.summary()

# ------------------------------------------------------------
# 7. AUTOENCODER
# ------------------------------------------------------------
print("\n7. Construyendo AUTOENCODER...")
>>>>>>> etapa2
autoencoder_input = layers.Input(shape=(784,), name='autoencoder_input')
x = encoder(autoencoder_input)
autoencoder_output = decoder(x)
autoencoder = models.Model(autoencoder_input, autoencoder_output, name='autoencoder')
<<<<<<< HEAD
autoencoder.summary()
=======
#autoencoder.summary()
>>>>>>> etapa2

# ------------------------------------------------------------
# 8. COMPILAR
# ------------------------------------------------------------
print("\n8. Compilando autoencoder...")
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mse']
)
print(" Autoencoder compilado")

# ------------------------------------------------------------
# 9. ENTRENAR
# ------------------------------------------------------------
print("\n9. Entrenando autoencoder...")
print("   Épocas: 50, Batch size: 256")
print("   Esto tomará varios minutos...\n")

history = autoencoder.fit(
    X, X,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

print(" Entrenamiento completado!")

# ------------------------------------------------------------
# 10. GUARDAR MODELOS
# ------------------------------------------------------------
print("\n10. Guardando modelos...")
encoder.save('encoder_fashion_mnist.h5')
autoencoder.save('autoencoder_completo.h5')
print(" Encoder guardado: encoder_fashion_mnist.h5")
print(" Autoencoder guardado: autoencoder_completo.h5")

# ------------------------------------------------------------
# 11. GENERAR VECTORES LATENTES
# ------------------------------------------------------------
print("\n11. Generando vectores latentes...")
X_latent = encoder.predict(X, verbose=1)
print(f" X_latent shape: {X_latent.shape}")

# ------------------------------------------------------------
<<<<<<< HEAD
# 12. VISUALIZAR UNA IMAGEN
=======
# 12. VISUALIZAR IMAGEN
>>>>>>> etapa2
# ------------------------------------------------------------
print("\n12. Visualizando imagen original...")
nro_de_imagen = 0
plt.figure(figsize=(6,6))
plt.imshow(X.values[nro_de_imagen,:].reshape(28,28), cmap='gray')
<<<<<<< HEAD
plt.title(f'Imagen Original - Fashion MNIST')
=======
plt.title('Imagen Original - Fashion MNIST')
>>>>>>> etapa2
plt.axis('off')
plt.savefig('imagen_original.png')
plt.show()

# ------------------------------------------------------------
<<<<<<< HEAD
# 13. BUSCAR IMÁGENES SIMILARES
# ------------------------------------------------------------
print("\n13. Buscando imágenes similares...")

query = X_latent[0:1]
similitudes = cosine_similarity(query, X_latent)[0]
indices_ordenados = np.argsort(similitudes)[::-1]

=======
# 13. IMÁGENES SIMILARES
# ------------------------------------------------------------
print("\n13. Buscando imágenes similares...")
query = X_latent[0:1]
similitudes = cosine_similarity(query, X_latent)[0]
indices_ordenados = np.argsort(similitudes)[::-1]
>>>>>>> etapa2
top_k = 5
indices_top = indices_ordenados[:top_k]

print(f"\nTop {top_k} imágenes más similares:")
plt.figure(figsize=(15, 4))
for i, idx in enumerate(indices_top):
    plt.subplot(1, top_k, i + 1)
    plt.imshow(X.values[idx,:].reshape(28,28), cmap='gray')
    plt.title(f'Sim: {similitudes[idx]:.3f}')
    plt.axis('off')
plt.suptitle('Imágenes más similares en espacio latente')
plt.savefig('imagenes_similares.png')
plt.show()

# ------------------------------------------------------------
<<<<<<< HEAD
# 14. FUNCIÓN PARA GENERAR IMÁGENES
# ------------------------------------------------------------
print("\n14. Definiendo función generar_imagen()...")

=======
# 14. GENERAR IMAGEN CON RUIDO
# ------------------------------------------------------------
print("\n14. Definiendo función generar_imagen()...")
>>>>>>> etapa2
def generar_imagen(vector_latente):
    if len(vector_latente.shape) == 1:
        vector_latente = vector_latente.reshape(1, -1)
    imagen_generada = decoder.predict(vector_latente, verbose=0)
    return imagen_generada[0]

print("✅ Función generar_imagen() lista")

<<<<<<< HEAD
# ------------------------------------------------------------
# 15. PROBAR GENERACIÓN CON RUIDO
# ------------------------------------------------------------
print("\n15. Probando generación con ruido...")

=======
print("\n15. Probando generación con ruido...")
>>>>>>> etapa2
query_vector = X_latent[0]
ruido = np.random.normal(0, 0.5, latent_dim)
query_variado = query_vector + ruido
img_variada = generar_imagen(query_variado)

plt.figure(figsize=(12, 4))
<<<<<<< HEAD

=======
>>>>>>> etapa2
plt.subplot(1, 3, 1)
plt.imshow(X.values[0,:].reshape(28,28), cmap='gray')
plt.title('Original')
plt.axis('off')
<<<<<<< HEAD

=======
>>>>>>> etapa2
plt.subplot(1, 3, 2)
plt.imshow(img_variada.reshape(28,28), cmap='gray')
plt.title('Con ruido')
plt.axis('off')
<<<<<<< HEAD

=======
>>>>>>> etapa2
plt.subplot(1, 3, 3)
plt.imshow(decoder.predict(query_vector.reshape(1,-1), verbose=0)[0].reshape(28,28), cmap='gray')
plt.title('Reconstruida')
plt.axis('off')
<<<<<<< HEAD

=======
>>>>>>> etapa2
plt.savefig('generacion_imagenes.png')
plt.show()

# ------------------------------------------------------------
# 16. COMPLETADO
# ------------------------------------------------------------
print("\n" + "="*60)
print("🎉 ETAPA 1 COMPLETADA EXITOSAMENTE!")
print("="*60)