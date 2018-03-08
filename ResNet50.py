from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.utils import plot_model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = Flatten()(x)
x = Dropout(rate=0.99)(x)
pred = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=pred)

print(model.summary())
#plot_model(model, to_file='resnet50.png')

for layer in base_model.layers[:147]:
    layer.trainable = False
for layer in base_model.layers[147:]:
    layer.trainable = True

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()
#Каждый класс должен быть в отдельной подпапке
train_generator = train_datagen.flow_from_directory(
     "D:\ML\ArtiD\kaggle\GoogleLandmark\\train_set",
    target_size=(224, 224),
    batch_size=50,
    class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch=100)