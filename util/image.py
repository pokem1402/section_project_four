import tensorflow as tf
import matplotlib.pyplot as plt
from util.util import is_notebook
import os

def load(input_file, target_file):
    return load_image(input_file), load_image(target_file)

def load_image(image_file):
    
    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)
    return input_image
    

def resize(input_image, target_image, height, width):
    input_image = resize_image(input_image, height, width)
    target_image = resize_image(target_image, height, width)
    
    return input_image, target_image

def resize_image(input_image, height, width):
    return tf.image.resize(input_image, [height, width],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def random_crop(input_image, target_image):
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, 256, 256, 3])

    return cropped_image[0], cropped_image[1]

def normalize_image(input_image):
    return (input_image / 127.5) - 1

def normalize(input_image, target_image):

    return normalize_image(input_image), normalize_image(target_image)


@tf.function()
def random_jitter(input_image, target_image):
    # resizing to 286 x 286 x 3
    input_image, target_image = resize(input_image, target_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        # random mirrorring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def load_image_train(input_file, target_file):
    input_image, target_image = load(input_file, target_file)
    input_image, target_image = random_jitter(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def load_image_test(input_file, target_file):
    input_image, target_image = load(input_file, target_file)
    input_image, target_image = resize(input_image, target_image, 256, 256)
    input_image, target_image = normalize(input_image, target_image)

    return input_image, target_image


def generate_images(model, test_input, tar, at,
                    save_fig=False, save_dir = "snapshot"):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            f'{save_dir}/snapshot at epoch {at}', bbox_inches='tight', pad_inches=0, dpi=100)

    if is_notebook():
        plt.show()


def generate_image(generator, input_image_path,
             target_image_path,
             only_predict,          # save prediction image only
             save_dir):
    
    image_file_name = os.path.basename(input_image_path)
    
    raw_input_image = load_image(input_image_path)
    input_image = resize_image(raw_input_image, 256, 256)
    input_image = normalize_image(input_image)
    
    gen_image = generator(input_image[tf.newaxis, ...], training=False)
        
    plt.figure(figsize = (15, 15))
       
    if only_predict: # TODO UN-normalize?
        tf.keras.utils.save_img(
            save_dir + "predct_" + image_file_name, gen_image[0]*0.5+0.5
        )
    else:
        display_list = [raw_input_image, gen_image[0]*0.5+0.5]
        title = ["Input Image", "Predicted Image"]
        
        if target_image_path is not None:
            raw_target_image = load_image(target_image_path)
            
            display_list.insert(1, raw_target_image)
            title.insert(1, "Ground Truth")
        
        n = len(display_list)
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i])
            plt.axis('off')
            
        plt.tight_layout()
        if is_notebook():
            plt.show()
        