import tensorflow as tf
import numpy as np

def decode_image(path, size=256):
    """ Load and resize the input texture """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    max_size = img.shape[0]
    if max_size > img.shape[1]: max_size = img.shape[1]

    img = tf.image.resize_with_crop_or_pad(img, max_size, max_size)
    img = tf.image.resize(img, [size, size]) 
    img = tf.image.convert_image_dtype(img, tf.float32) / 255

    return img[None]

def decode_image_with_patch(path, patch_size= 32, target_size=256, pad_type='wrap'):
    """ Load and resize the input texture 
    pad type : see np.pad: 'wrap','reflect','symmetric' etc..."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    max_size = img.shape[0]
    if max_size > img.shape[1]: max_size = img.shape[1]
    pad_size = int(np.round((target_size-patch_size)/2))

    img = tf.image.resize_with_crop_or_pad(img, patch_size, patch_size)
    img = np.pad(img,pad_width=((pad_size,pad_size),(pad_size,pad_size),(0,0)), mode=pad_type)
    img = tf.image.resize(img, [target_size, target_size]) 

    img = tf.image.convert_image_dtype(img, tf.float32)/255

    return img[None]

def decode_image_spatial_tag(path, size=256,channels=3,method = 'nearest'):
    """ Load and resize the input texture """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, [size, size], method = method) 
    img = tf.math.round(img)
    img = tf.image.convert_image_dtype(img, tf.float32)/255
    return img[None]