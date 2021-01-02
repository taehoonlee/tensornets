from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow.compat.v1 as tf
import tensornets as nets

from imagenet_preprocessing import input_fn as _input_fn
from tensorflow import contrib


tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 50,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'batch_size', 200, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'steps', None, 'The number of steps for evaluation.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'model_name', 'ResNet50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/taehoonlee/Data/imagenet/tfrecords',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'The eval image size')

tf.app.flags.DEFINE_integer(
    'normalize', 0, 'The normalization type')

FLAGS = tf.app.flags.FLAGS


# Simple trick to suppress the warning
# "It seems that global step has not been increased"
class hook(tf.train.StepCounterHook):
    def __init__(self, every_n_steps):
        self._steps = 0
        super(hook, self).__init__(every_n_steps)

    def after_run(self, run_context, run_values):
        self._steps += 1
        if self._timer.should_trigger_for_step(self._steps):
            t, steps = self._timer.update_last_triggered_step(self._steps)
            if t is not None:
                tf.logging.info("%g secs per step on average", t / steps)


def input_fn():
    return _input_fn(
        is_training=False,
        data_dir=FLAGS.dataset_dir,
        batch_size=FLAGS.batch_size,
        eval_image_size=FLAGS.eval_image_size,
        normalize=FLAGS.normalize)


def model_fn(features, labels, mode):
    models = []
    logits = []
    classes = []
    init_op = [tf.train.get_or_create_global_step().initializer]
    for (i, model_name) in enumerate(FLAGS.model_name.split(',')):
        with tf.device("/gpu:%d" % i):
            network_fn = getattr(nets, model_name)
            models.append(network_fn(features, is_training=False))
            logits.append(models[i].get_outputs()[-2])
            classes.append(tf.argmax(logits[i], axis=1))
            if FLAGS.checkpoint_path is None:
                init_op.extend(models[i].pretrained())

    scaffold = None
    if FLAGS.checkpoint_path is None:
        scaffold = tf.train.Scaffold(init_op=init_op)

    loss = []
    for i in range(len(models)):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits[i], labels=labels)
        loss.append(cross_entropy)
    loss = tf.reduce_sum(loss)

    metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}
        for i in range(len(models)):
            top1 = tf.metrics.accuracy(labels=labels, predictions=classes[i])
            top5 = contrib.metrics.streaming_sparse_recall_at_k(
                logits[i], tf.cast(labels, tf.int64), k=5)
            size = sum([w.shape.num_elements()
                        for w in models[i].get_weights()])
            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'none'
            flops = tf.profiler.profile(tf.get_default_graph(),
                                        run_meta=run_meta, options=opts)
            metrics.update({"%dTop1" % i: top1,
                            "%dTop5" % i: top5,
                            "%dMAC" % i: (tf.constant(flops.total_float_ops), tf.no_op()),
                            "%dSize" % i: (tf.constant(size), tf.no_op())})

    return tf.estimator.EstimatorSpec(
        mode=mode,
        scaffold=scaffold,
        predictions=None,
        loss=loss,
        train_op=None,
        eval_metric_ops=metrics,
        export_outputs=None)


def main(argv=None):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory.')

    tf.logging.set_verbosity(tf.logging.INFO)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.checkpoint_path)

    if FLAGS.steps is None:
        FLAGS.steps = 50000 // FLAGS.batch_size

    results = classifier.evaluate(
        input_fn=input_fn, steps=FLAGS.steps,
        hooks=[hook(every_n_steps=FLAGS.log_every_n_steps)])

    print("| {:5d} Samples    | Top-1       | Top-5       | MAC    | Size   |".format(
          FLAGS.batch_size * FLAGS.steps))
    print("|------------------|-------------|-------------|--------|--------|")
    for (i, model_name) in enumerate(FLAGS.model_name.split(',')):
        print("| {:16s} | {:6.3f}      | {:6.3f}      | {:5.1f}M | {:5.1f}M |".format(
              model_name.split('Net')[-1] if len(model_name) > 16 else model_name,
              100 * (results["%dTop1" % i]),
              100 * (results["%dTop5" % i]),
              results["%dMAC" % i] / 10e5,
              results["%dSize" % i] / 10e5))


if __name__ == '__main__':
    main(sys.argv[1:])
