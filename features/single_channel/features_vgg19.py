import numpy as np
import tensorflow
from tensorflow.keras.applications.vgg19 import (preprocess_input, VGG19)
from tensorflow.keras.preprocessing.image import (img_to_array, load_img)


def extract_features_vgg19(file_name: str,
                           layer_index: list
                           ) -> np.ndarray:
    """
    Extract features from the specified layers of a pre-trained VGG19 model using the input image.

    :param file_name: Figure name of the spectogram, supports .png and .jpg
    :param layer_index: List of integers indicating the indices of the layers to extract features from.

    Notes:
    -------
    Reads image from current working directory.

    :return: Flattened feature maps extracted from the input image.

    Example:
    --------
    >>> from_path = 'C:/path/to/file/'
    >>> file_name = 'image.png'
    >>> layer_index = [3, 8, 13]
    >>> features = extract_features_vgg19(from_path, file_name, layer_index)
    """

    # Load the pre-trained VGG19 model
    model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Load the input image and preprocess it for the VGG19 model
    img = load_img(file_name, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features from the input image
    feature_maps = model.predict(img)

    # Flatten the feature maps from the specified layers
    flattened_features = np.concatenate(

        [feature_maps[idx].flatten() for idx in layer_index]

    )

    return flattened_features
