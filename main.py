%matplotlib inline

import os
# Set environment variable to allow GPU memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# if using Theano with GPU
# os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

import tensorflow_datasets as tfds
import tensorflow as tf

## Importação do dataset
train_data, val_data, test_data = tfds.load('cats_vs_dogs', split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'], shuffle_files=True, as_supervised=True)

## Resumo
print('Exemplos de treinamento =', len(list(train_data)))
print('Exemplos de validação =', len(list(val_data)))
print('Exemplos de teste =', len(list(test_data)))

## 3 exemplos do dataset
for data in train_data.take(3):
    image, label = data

    print("Classe: {}".format(label))
    plt.imshow(image)
    plt.show()

## Transfer learning (VGG16 do Keras treinado com ImageNet)
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()

'''
congelar pesos da rede mantendo todas as camadas, exceto 
a última que será substituída por uma camada softmax
'''
num_classes = 2

# make a reference to VGG's input layer
inp = vgg.input

# make a new softmax layer with num_classes neurons
new_classification_layer = Dense(num_classes, activation='softmax')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(vgg.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)

## compilação e resumo da rede
# make all layers untrainable by freezing weights (except for last layer)
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

# ensure the last layer is trainable/not frozen
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

'''
Retreinamento da rede: rótulos classificados pelo método one-hot encoding,
imagens são redimensionadas para 224x224 com valores de cada pixel entre 0 e 1
com valores convertidos para float de 32 bits.
É utilizado lotes de 128 elementos.
'''
# Dimensão de imagens usadas pela VGG
IMAGE_SIZE = (224, 224)

# Função utilizada para redimensionar e normalizar as imagens
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    label = tf.one_hot(label, num_classes)  # Converte o rótulo para "one-hot encoding"
    return  image, label

# Define o tamanho do lote de dados de treinamento e validação
BATCH_SIZE = 128

# Cria lotes de dados usando o método map() para chamar a função format_image()
train_batches = train_data.map(format_image).batch(BATCH_SIZE)
test_batches = test_data.map(format_image).batch(BATCH_SIZE)
val_batches = val_data.map(format_image).batch(BATCH_SIZE)

# Verifica e ajusta os tipos de dados para evitar problemas no treinamento
train_batches = train_batches.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
val_batches = val_batches.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
test_batches = test_batches.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

for images, labels in train_batches.take(1):
    print(f"Shape das imagens: {images.shape}")
    print(f"Shape dos rótulos: {labels.shape}")

# Ajusta os buffers de pré-carregamento para desempenho
AUTOTUNE = tf.data.AUTOTUNE
train_batches = train_batches.prefetch(buffer_size=AUTOTUNE)
val_batches = val_batches.prefetch(buffer_size=AUTOTUNE)
test_batches = test_batches.prefetch(buffer_size=AUTOTUNE)

# Treinamento
history2 = model_new.fit(
    train_batches,
    epochs=10,
    validation_data=val_batches,
)

## Mostrar/plotar a evolução da função de custo e da acurácia ao longo das épocas
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)

ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history2.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

## Validação final nos dados de teste
loss, accuracy = model_new.evaluate(test_batches, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)

## Predição para uma imagem de entrada (cachorro ou gato no exemplo)
from tensorflow.keras.utils import load_img, img_to_array

# Carregamento e pré-processamento da imagem
img_path = './imagem1.jpg'
img = load_img(img_path, target_size=IMAGE_SIZE)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Mostrar a imagem
plt.imshow(img)
plt.axis('off')  # Ocultar os eixos
plt.title("Imagem de entrada")
plt.show()

# Predição
predictions = model_new.predict(img_array)
predicted_class = np.argmax(predictions)  # Classe com maior probabilidade (0 = gato, 1 = cachorro)
probability = predictions[0][predicted_class]  # Probabilidade da classe predita

# Resultados
classes = ["Gato", "Cachorro"]
print(f"Classe predita: {classes[predicted_class]}")
print(f"Probabilidade: {probability:.2f}")
    