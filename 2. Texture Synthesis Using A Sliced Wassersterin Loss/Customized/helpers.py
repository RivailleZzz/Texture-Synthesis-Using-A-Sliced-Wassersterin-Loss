import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def initialize_image(SIZE, texture):
  image = np.zeros((1,SIZE,SIZE,3))
  image = image + tf.reduce_mean(texture, axis=(1, 2))[None, None]
  image = image + (tf.random.normal((1, SIZE,SIZE, 3))*1e-2)
  return image

def gram_loss(image, target, extractor):
  #extractor(image) is list of (1,X,Y,C)
  gram_image = [gram_matrix(feature) for feature in extractor(image)]
  gram_target = [gram_matrix(feature) for feature in extractor(target)]
  #sum of MSE gram matrix difference across layers
  style_loss = tf.add_n([tf.reduce_mean((gram_image[idx]-gram_target[idx])**2) for idx,_ in enumerate(gram_image)])
  return style_loss

def slice_loss(image, target, extractor):
  image_feats = extractor(image)
  target_feats = extractor(target)
  losses =[tf.reduce_mean((output-target_feats[i])**2) 
                          for i, output in enumerate(image_feats)]
  loss = tf.add_n(losses)
  return loss
