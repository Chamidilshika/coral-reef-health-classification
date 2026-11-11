from tensorflow.keras.processing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range = 20,
    horizontal_flip = True,
    zoom_range = 0.2,
    brightness_range = [0.8, 1.2]
)