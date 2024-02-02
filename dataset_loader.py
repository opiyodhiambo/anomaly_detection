import math
import numpy as np
import matplotlib.pyplot as plt
import tensoflow as tf


def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image


def load_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    image_paths = [f"{image_dir}/{filename}" for filename in df['filename']]
    labels = df['class'].values

    return image_paths, labels


def create_tf_dataset(csv_path, image_dir, batch_size=32):
    image_paths, labels = load_dataset(csv_path, image_dir)
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (process_image(x), y))
    dataset = dataset.batch(batch_size)

    return dataset
