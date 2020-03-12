"""Weight translation of MobileNetv3 variants
(tested with tensornets: 0.4.3 and tensorflow: 1.15.0)

The codes are executable on the path "research/slim/"
in the "tensorflow/models" repository.

For the 0.75 variants, the following modifications are necessary.

In the line 116 of "research/slim/nets/mobilenet/mobilenet.py",
 def op(opfunc, multiplier_func=depth_multiplier, **params):
   multiplier = params.pop('multiplier_transform', multiplier_func)
-  return _Op(opfunc, params=params, multiplier_func=multiplier)
+  if params.get('normalizer_fn', True) is not None:
+    return _Op(opfunc, params=params, multiplier_func=multiplier)
+  else:
+    return _Op(opfunc, params=params, multiplier_func=lambda x, y: x)
"""
import numpy as np
import tensorflow as tf
import tensornets as nets

from datasets import imagenet
from nets.mobilenet import mobilenet_v3

models_list = [
    (nets.MobileNet75v3large, (224, 224, 3), 'mobilenet_75_v3_large',
     mobilenet_v3.large, 0.75, 'v3-large_224_0.75_float/ema/model-220000'),
    (nets.MobileNet75v3small, (224, 224, 3), 'mobilenet_75_v3_small',
     mobilenet_v3.small, 0.75, 'v3-small_224_0.75_float/ema/model-497500'),
    (nets.MobileNet100v3large, (224, 224, 3), 'mobilenet_100_v3_large',
     mobilenet_v3.large, 1.0, 'v3-large_224_1.0_float/ema/model-540000'),
    (nets.MobileNet100v3small, (224, 224, 3), 'mobilenet_100_v3_small',
     mobilenet_v3.small, 1.0, 'v3-small_224_1.0_float/ema/model-388500'),
    (nets.MobileNet100v3largemini, (224, 224, 3),
     'mobilenet_100_v3_large_mini',
     mobilenet_v3.large_minimalistic, 1.0,
     'v3-large-minimalistic_224_1.0_float/ema/model-342500'),
    (nets.MobileNet100v3smallmini, (224, 224, 3),
     'mobilenet_100_v3_small_mini',
     mobilenet_v3.small_minimalistic, 1.0,
     'v3-small-minimalistic_224_1.0_float/ema/model-498000'),
]


for (net, shape, model_name, net_slim, alpha, checkpoint) in models_list:

    with tf.Graph().as_default():

        inputs = tf.compat.v1.placeholder(tf.float32, [None] + list(shape))

        with tf.contrib.slim.arg_scope(mobilenet_v3.training_scope(is_training=False)):
            logits, endpoints = net_slim(inputs, depth_multiplier=alpha)

        saver = tf.compat.v1.train.Saver()

        weights_tfslim = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        model = net(inputs, scope='a')

        img = nets.utils.load_img('/home/taehoonlee/tensornets/cat.png',
                                  target_size=int(shape[0] * 8 / 7),
                                  crop_size=shape[0])

        with tf.compat.v1.Session() as sess:

            # Retrieve values
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, checkpoint)
            names = [w.name[2:] for w in model.weights()]
            values = sess.run(weights_tfslim)

            # Trim the background class (1001 -> 1000)
            for i in range(-2, 0):
                values[i] = np.delete(np.squeeze(values[i]), 0, axis=-1)

            # Save the values as the TensorNets format
            np.savez(model_name, names=names, values=values)

            # Load and set the values
            weights = model.weights()
            values = nets.utils.parse_weights(model_name + '.npz')
            sess.run([w.assign(v) for (w, v) in zip(weights, values)])

            # Run equivalence tests
            preds, preds_tfslim = sess.run([model, endpoints['Predictions']],
                                           {inputs: model.preprocess(img)})
            preds_tfslim = preds_tfslim[:, 1:]
            np.testing.assert_allclose(preds, preds_tfslim, atol=2e-4)
            print(model_name, 'ok')
