

**Autor:** Monica Fernandez 
**Fecha:** 12/02/2026  

---

##  Descripción del Proyecto

Este proyecto implementa **Transfer Learning** en dos etapas utilizando el dataset **Fashion MNIST**.

###  Etapa 1: Autoencoder
- Arquitectura: 784 → 128 → 64 → 32 → 64 → 128 → 784
- Dimensión latente: 32
- Entrenamiento: 50 épocas, batch_size=256
- Función de pérdida: binary_crossentropy

###  Etapa 2: Clasificador con Transfer Learning
- Encoder de Etapa 1 **congelado** (trainable=False)
- Capas añadidas: Dense(64) → Dense(32) → Dense(10)
- Entrenamiento: 15 épocas, batch_size=256


---


