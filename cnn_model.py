import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_cnn_model(upload_folder, learning_rate, epochs):
    # Charger les images depuis le dossier uploadé
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Générateur d'images d'entraînement
    train_generator = datagen.flow_from_directory(
        upload_folder,
        target_size=(150, 150),  # Adapter à la taille souhaitée
        batch_size=32,
        class_mode='binary'  # Modifier selon la tâche (binaire ou catégorielle)
    )

    # Vérification qu'il y a des images à entraîner
    if train_generator.samples == 0:
        raise ValueError("Aucune image trouvée dans le dossier pour l'entraînement.")

    # Construction du modèle CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Pour une classification binaire
    ])

    # Compilation du modèle
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(train_generator, epochs=epochs)

    # Évaluation du modèle et retour de la précision
    accuracy = model.evaluate(train_generator)[1]
    return accuracy
