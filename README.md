# Classification-Neural-Network

## Установка зависимостей

``` sh
!pip install scipy
!pip install pandas
```

## Подключение библиотек и прочих зависимостей

``` python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import PIL
import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt
```

### Создание модели

Использование функции активации "RELU". А также при помощи слоёв Conv2D и MaxPolling2D происходит свертка изображения

``` python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

...

model.add(Flatten(input_shape=(150, 150, 3)))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))

model.summary()
```

### Обучение и сохранение модели

``` python
model.fit(x = training_set , validation_data = valid_set , epochs = 100, batch_size = 100)
model.save_weights('model.weights.h5')
```

### Итоговые графики

Accuracy
![image](https://github.com/user-attachments/assets/8eb5a899-8564-44af-bb39-5cdd7f09704b)

Loss
![image](https://github.com/user-attachments/assets/a1a76784-2181-4f2e-9fcb-c4836ddabdd9)



