

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models

print("="*60)
print("ETAPA 2: CLASIFICADOR CON TRANSFER LEARNING")
print("="*60)

# ------------------------------------------------------------
# 1. CARGAR DATOS
# ------------------------------------------------------------
print("\n1. Cargando Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Aplanar imágenes (784 píxeles)
X_train_flat = x_train.reshape(x_train.shape[0], -1)
X_test_flat = x_test.reshape(x_test.shape[0], -1)

# Crear DataFrames
data = pd.DataFrame(X_train_flat)
data_test = pd.DataFrame(X_test_flat)

# Insertar etiquetas
data.insert(0, 'label', y_train)
data_test.insert(0, 'label', y_test)

# Separar X y y
X = data.iloc[:, 1:]
X_test = data_test.iloc[:, 1:]
y = data.iloc[:, 0]
y_test = data_test.iloc[:, 0]

# Normalizar
X = X.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"✅ X_train: {X.shape}")
print(f"✅ X_test: {X_test.shape}")

# ------------------------------------------------------------
# 2. CARGAR ENCODER PRETRAINADO
# ------------------------------------------------------------
print("\n2. Cargando encoder pre-entrenado...")
encoder = keras.models.load_model('encoder_fashion_mnist.h5')

# CONGELAR ENCODER
encoder.trainable = False
print("✅ Encoder CONGELADO (no se entrena)")

# ------------------------------------------------------------
# 3. CONSTRUIR CLASIFICADOR
# ------------------------------------------------------------
print("\n3. Construyendo clasificador...")

clasificador_input = keras.Input(shape=(32,), name='clasificador_input')
x = layers.Dense(64, activation='relu', name='clasificador_dense_1')(clasificador_input)
x = layers.Dense(32, activation='relu', name='clasificador_dense_2')(x)
clasificador_output = layers.Dense(10, activation='softmax', name='clasificador_output')(x)

clasificador = models.Model(clasificador_input, clasificador_output, name='clasificador')
clasificador.summary()

# ------------------------------------------------------------
# 4. MODELO COMPLETO
# ------------------------------------------------------------
print("\n4. Construyendo modelo completo...")

modelo_input = keras.Input(shape=(784,), name='modelo_input')
latent_vector = encoder(modelo_input)
modelo_output = clasificador(latent_vector)

modelo_final = models.Model(modelo_input, modelo_output, name='fashion_mnist_classifier')
modelo_final.summary()

# ------------------------------------------------------------
# 5. COMPILAR
# ------------------------------------------------------------
print("\n5. Compilando modelo...")
modelo_final.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("✅ Modelo compilado")

# ------------------------------------------------------------
# 6. ENTRENAR (SOLO CLASIFICADOR)
# ------------------------------------------------------------
print("\n6. Entrenando clasificador (encoder congelado)...")
print("   Épocas: 15, Batch size: 256\n")

history = modelo_final.fit(
    X, y,
    epochs=15,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# ------------------------------------------------------------
# 7. EVALUAR
# ------------------------------------------------------------
print("\n7. Evaluando modelo...")
test_loss, test_acc = modelo_final.evaluate(X_test, y_test, verbose=0)
print(f"\n **PRECISIÓN EN TEST: {test_acc:.4f} ({test_acc*100:.2f}%)**")

# ------------------------------------------------------------
# 8. GUARDAR MODELO
# ------------------------------------------------------------
print("\n8. Guardando modelo...")
modelo_final.save('clasificador_fashion_mnist.h5')
print(" Modelo guardado: clasificador_fashion_mnist.h5")

# ------------------------------------------------------------
# 9. PROBAR PREDICCIONES
# ------------------------------------------------------------
print("\n9. Probando predicciones...")

class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Tenis', 'Bolsa', 'Botín']

print("\n" + "-"*60)
print("PREDICCIONES (10 PRIMERAS IMÁGENES DE TEST)")
print("-"*60)

correctas = 0
for i in range(10):
    img = X_test.values[i:i+1]
    pred = modelo_final.predict(img, verbose=0)
    pred_class = np.argmax(pred[0])
    real_class = y_test.values[i]
    
    marca = "✓" if pred_class == real_class else "✗"
    if pred_class == real_class:
        correctas += 1
    
    print(f"{i+1:2d}. Real: {class_names[real_class]:10s} → Pred: {class_names[pred_class]:10s} {marca}")

print("-"*60)
print(f"\n📈 Precisión en muestra: {correctas}/10 ({correctas*10}%)")

# ------------------------------------------------------------
# 10. VISUALIZAR PREDICCIONES
# ------------------------------------------------------------
print("\n10. Visualizando predicciones...")

plt.figure(figsize=(15, 8))
for i in range(8):
    img = X_test.values[i].reshape(28, 28)
    pred = modelo_final.predict(X_test.values[i:i+1], verbose=0)
    pred_class = np.argmax(pred[0])
    real_class = y_test.values[i]
    
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray')
    color = 'green' if pred_class == real_class else 'red'
    plt.title(f'Real: {class_names[real_class]}\nPred: {class_names[pred_class]}', color=color)
    plt.axis('off')

plt.suptitle(f'Predicciones - Precisión Test: {test_acc*100:.2f}%', fontsize=14)
plt.tight_layout()
plt.savefig('predicciones_fashion_mnist.png')
plt.show()

# ------------------------------------------------------------
# 11. COMPLETADO
# ------------------------------------------------------------
print("\n" + "="*60)
print(" ETAPA 2 COMPLETADA EXITOSAMENTE!")
print("="*60)
print(f"\n PRECISIÓN FINAL: {test_acc*100:.2f}%")
print("\n Archivos generados:")
print("   - clasificador_fashion_mnist.h5")
print("   - predicciones_fashion_mnist.png")