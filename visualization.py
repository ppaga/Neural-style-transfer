import os, sys, pprint, time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from skimage.transform import resize
from matplotlib.pyplot import imshow, imread

from keras.applications import vgg19

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("iterations", 50, "number of iterations [10]")
flags.DEFINE_integer("iteration_counter", 10, "display frequency [10]")
flags.DEFINE_float("learning_rate", 1., "initial Learning rate for adam [10.]")
flags.DEFINE_float("decay_rate", 1., "learning rate decay per thousand updates - set to 1. for constant learning rate [1.]")
flags.DEFINE_float("beta1", 0.99, "Momentum term of adam [0.99]")
flags.DEFINE_float("alpha", 1., "content weight")
flags.DEFINE_integer("width", 256, "width of the picture [256]")
flags.DEFINE_integer("height", 256, "height of the picture [256]")
flags.DEFINE_integer("style_width", 256, "width of the style picture [256]")
flags.DEFINE_integer("style_height", 256, "height of the style picture [256]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("result_path", "./result.png", "path of the resulting image")
flags.DEFINE_string("content_path", "./content.jpg", "path of the content image")
flags.DEFINE_string("style_path", "./style.jpg", "path of the content image")


FLAGS = flags.FLAGS
class vgg_net():
    def __init__(self, shape, var_dict=None):
        self.vgg = vgg19.VGG19(include_top=False,input_shape=shape)
        layers = self.vgg.layers
        self.layer_dict = {}
        if var_dict is None:
            for layer in self.vgg.layers:
                layer.trainable=False
                if 'conv' in layer.name:
                    name = layer.name+'_kernel'
                    weights = layer.get_weights()[0]
                    init = tf.constant_initializer(weights)
                    self.layer_dict[name] = tf.get_variable(name, shape = weights.shape, initializer = init, trainable=False)
                    
                    name = layer.name+'_bias'
                    weights = layer.get_weights()[1]
                    init = tf.constant_initializer(weights)
                    self.layer_dict[name] = tf.get_variable(name, shape = weights.shape, initializer = init, trainable=False)
        else:
            self.layer_dict = var_dict
    def conv(self,x, weights, bias):
        y = tf.nn.conv2d(x, filter = weights, strides = [1,1,1,1],padding = 'SAME')
        y = tf.nn.bias_add(y, bias)
        z = tf.nn.relu(y)
        return y,z
    def features(self,x):
        y = 255*x
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=y)
        y = tf.concat(axis=3, values=[blue - 103.939,green - 116.779,red - 123.68])
        outputs = []
        for layer in self.vgg.layers:
            if 'conv' in layer.name:
                feature,y = self.conv(y,self.layer_dict[layer.name+'_kernel'],self.layer_dict[layer.name+'_bias'])
                outputs.append(feature/255)
            if 'pool' in layer.name:
                y = tf.nn.max_pool(y, ksize=[1,2, 2,1], strides=[1,2,2,1], padding='SAME')
        return outputs
def content_loss_func(features_x, features_y):
    features = list(zip(features_x, features_y))
    loss = 0
    for fx,fy in features:
        loss += tf.reduce_sum(tf.square(fx-fy))/2.
    return loss
def style_loss_func(Grams_x, Grams_y):
    Grams = list(zip(Grams_x, Grams_y))
    loss = 0
    for Gram_x,Gram_y in Grams:
        loss += tf.losses.mean_squared_error(Gram_x, Gram_y)/4.
    return loss
def Gram(features):
    Grams = []
    for feature in features:
        y = tf.squeeze(feature)
        shape = tf.shape(y)
        y = tf.reshape(y,[tf.reduce_prod(shape[:2]),-1])
        z = tf.matmul(y,y,transpose_a=True)/tf.to_float(tf.reduce_prod(shape[:2]))
        Grams.append(z)
    return Grams

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
#    to speed up the computation, we compute the features of the content and style pictures first
    content_shape = [FLAGS.width,FLAGS.height]
    style_shape = [FLAGS.style_width,FLAGS.style_height]
    content_image = np.expand_dims(resize(imread(FLAGS.content_path), content_shape), axis=0).astype(float)
    if len(content_image.shape)==3:
        content_image = np.tile(content_image, [3,1,1])
        content_image = np.transpose(content_image, [1,2,0])
        content_image = np.expand_dims(content_image, axis=0)
    style_image = np.expand_dims(resize(imread(FLAGS.style_path), style_shape), axis=0).astype(float)
    
    # with tf.device('/GPU:0'):
    content_shape = content_shape + [FLAGS.c_dim,]
    style_shape = style_shape + [FLAGS.c_dim,]
    
    vgg = vgg_net(content_shape)
    var_dict = vgg.layer_dict
        
    vgg_style = vgg_net(style_shape, var_dict)

    content = tf.placeholder(tf.float32, [1,] + content_shape, name='content')
    style = tf.placeholder(tf.float32, [1,] + style_shape, name='style')

    content_features = vgg.features(content)
    
    style_features = vgg_style.features(style)
    Grams_style = Gram(style_features)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        CF, GS = sess.run([content_features, Grams_style], feed_dict = {content: content_image, style: style_image})
        sess.close()
    
#    once this is done, we can introduce the image proper and optimize it
    image_shape = [1,]+content_shape
    im_var = tf.Variable(tf.random_normal(image_shape), name='image', trainable=True)
#    If I use a simple sigmoid, the image values have to be quite high to reach 0 and 1, or close to them. So instead I stretch the sigmoid a bit
    image = tf.clip_by_value(tf.nn.sigmoid(im_var)*1.01-0.005*tf.ones_like(im_var), 0.,1.)
    
    image_features_content = vgg.features(image)
    image_features_style = vgg.features(image)
    grams_image = Gram(image_features_style)
    
    CF_const = []
    GS_const = []
    for feature, gram in list(zip(CF, GS)):
        CF_const.append(tf.constant(feature))
        GS_const.append(tf.constant(gram))

    # we follow the original Gatys et al. paper which used specific layers:        
    content_loss = content_loss_func(CF_const[8:9]+CF_const[14:15],image_features_content[8:9]+image_features_content[14:15])
    style_loss = style_loss_func([GS_const[0],GS_const[2],GS_const[4],GS_const[8],GS_const[12]],  [grams_image[0],grams_image[2],grams_image[4],grams_image[8],grams_image[12]])
    
    alpha = FLAGS.alpha
    loss = alpha*content_loss + style_loss
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    decay_rate = FLAGS.decay_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps = 1e3, decay_rate = decay_rate, staircase=False)

    optim = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1).minimize(loss, var_list = [im_var,], global_step = global_step)
    
    with tf.name_scope('summaries/'):
        tf.summary.scalar('log_loss', tf.log(loss))
        tf.summary.scalar('log_content', tf.log(content_loss))
        tf.summary.scalar('log_style', tf.log(style_loss))
        tf.summary.image('image', image)
        tf.summary.image('content', tf.constant(content_image))
        tf.summary.image('style', tf.constant(style_image))
    merged = tf.summary.merge_all()
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
#        # if a tensorboard log is needed, uncomment  the following lines and the ones in the iteration loop      
#        logs_dir = './logs/'
#        tl.files.exists_or_mkdir(logs_dir)
#        
#        #creates the tensorboard log writer
#        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())
        
        iter_counter = 0
        start_time = time.time()
        
        for iteration in range(FLAGS.iterations):
            # train the network and generate summary statistics for tensorboard
            _ = sess.run(optim)
            if iteration % FLAGS.iteration_counter ==0:
#                summary = sess.run(merged)
                print("iteration: [%3d/%3d]" % (iteration, FLAGS.iterations))
#                writer.add_summary(summary, iteration)
        img = sess.run(image)  
        tl.visualize.save_image(np.squeeze(img), FLAGS.result_path)
        print(time.time()-start_time)
if __name__ == '__main__':
    tf.app.run()
