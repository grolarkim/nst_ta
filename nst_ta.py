import tensorflow as tf

import PIL
import numpy as np

style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')

# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image

# Function to run style prediction on preprocessed style image.
# Run style transform on preprocessed style image
def run_style_transform(preprocessed_style_image, preprocessed_content_image):
    # Load the predict model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)
    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()

    # Load the transform model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)
    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()
    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()
    return stylized_image



def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



def style_transfer(title,content_src,style_src):
    title = title + ".png"
    content_name = content_src.split("/")[-1]
    content_path = tf.keras.utils.get_file(content_name, content_src)
    style_name = style_src.split("/")[-1]
    style_path = tf.keras.utils.get_file(style_name, style_src)
    # Load the input images.
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    # Preprocess the input images.
    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)
    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(preprocessed_style_image, preprocessed_content_image)
    tensor_to_image(stylized_image).save(title)


if __name__=="__main__":
    content_src = 'https://d1txao2peb1gkd.cloudfront.net/1210_19_50_37.jpg'
    style_src = 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg'
    style_transfer("내사진",content_src,style_src)
