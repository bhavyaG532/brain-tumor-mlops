import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

def load_data(data_dir):

    train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # ✅ Save class names BEFORE map
    class_names = train.class_names

    # ✅ Normalize
    train = train.map(lambda x, y: (x / 255.0, y))
    val = val.map(lambda x, y: (x / 255.0, y))

    return train, val, class_names
