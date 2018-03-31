import numpy as np
import tensorflow as tf
import tensornets as nets

from tensornets.datasets import imagenet
data_dir = '/home/taehoonlee/Data/imagenet/inputs'


def imagenet_load(data_dir, resize_wh, crop_wh, crops):
    return imagenet.load(
        data_dir, 'val', batch_size=10 if crops == 10 else 100,
        resize_wh=resize_wh,
        crop_locs=10 if crops == 10 else 4,
        crop_wh=crop_wh)


def test(models_list, crops=1, verbose=False):
    batches1 = imagenet_load(data_dir, 256, 224, crops)
    batches2 = imagenet_load(data_dir, 341, 299, crops)
    batches3 = imagenet_load(data_dir, 378, 331, crops)
    inputs, models, shapes, params = [], [], [], []
    labels, preds_list = [], []
    if verbose:
        print("")

    with tf.Graph().as_default():
        for (_net, _shape, _gpu) in models_list:
            with tf.device("gpu:%d" % _gpu):
                _input = tf.placeholder(tf.float32, [None] + list(_shape))
                _model = _net(_input, is_training=False)
                _weights = _model.get_weights()
                inputs.append(_input)
                models.append(_model)
                shapes.append(_shape)
                params.append(sum([w.shape.num_elements() for w in _weights]))

        with tf.Session() as sess:
            nets.pretrained(models)
            while True:
                try:
                    batch1, label1 = batches1.next()
                    batch2, label2 = batches2.next()
                    batch3, label3 = batches3.next()
                except:
                    break
                feed_dict = dict((i, m.preprocess(batch1 if s[0] == 224 else
                                                  batch2 if s[0] == 299 else
                                                  batch3))
                                 for (i, m, s) in zip(inputs, models, shapes))
                preds = sess.run(models, feed_dict)
                if crops > 1:
                    preds = [np.mean(pred.reshape(-1, crops, 1000), axis=1)
                             for pred in preds]
                labels.append(label1)
                preds_list.append(preds)
                if verbose:
                    print('.'),
        labels = np.concatenate(labels)

    if verbose:
        print("")

    def err(x):
        return 100 * (1 - sum(x) / float(len(x)))

    print("Crops: %d" % crops)
    print("Samples: %d" % len(labels))
    print("|                  | Top-1 | Top-5 | Top-1  | Top-5  | Size  |")
    print("|------------------|-------|-------|--------|--------|-------|")

    for i in range(len(models)):
        preds = np.concatenate([np.argsort(pred[i], axis=1)[:, -5:]
                                for pred in preds_list], axis=0)
        actuals = labels[:preds.shape[0]]
        top1 = (actuals == preds[:, -1])
        top5 = [1 if actual in pred else 0
                for (actual, pred) in zip(actuals, preds)]
        print("| %16s | %5d | %5d | %2.3f | %2.3f | %.1fM |" %
              (models[i].aliases[0][:16],
               sum(top1), sum(top5),
               err(top1), err(top5),
               params[i] / 10e5))


test([(nets.ResNet50, (224, 224, 3), 0),
      (nets.ResNet101, (224, 224, 3), 0),
      (nets.ResNet152, (224, 224, 3), 0),
      (nets.ResNeXt50, (224, 224, 3), 0),
      (nets.ResNeXt101, (224, 224, 3), 1),
      (nets.ResNeXt101c64, (224, 224, 3), 1),
      (nets.WideResNet50, (224, 224, 3), 1)])

test([(nets.ResNet50v2, (299, 299, 3), 0),
      (nets.ResNet101v2, (299, 299, 3), 1),
      (nets.ResNet152v2, (299, 299, 3), 1),
      (nets.ResNet200v2, (224, 224, 3), 0)])

test([(nets.Inception1, (224, 224, 3), 0),
      (nets.Inception2, (224, 224, 3), 1),
      (nets.Inception3, (299, 299, 3), 0),
      (nets.Inception4, (299, 299, 3), 0),
      (nets.InceptionResNet2, (299, 299, 3), 1)])

test([(nets.NASNetAlarge, (331, 331, 3), 0)])

test([(nets.NASNetAmobile, (224, 224, 3), 0),
      (nets.VGG16, (224, 224, 3), 0),
      (nets.VGG19, (224, 224, 3), 1),
      (nets.SqueezeNet, (224, 224, 3), 1)])

test([(nets.DenseNet121, (224, 224, 3), 0),
      (nets.DenseNet169, (224, 224, 3), 0),
      (nets.DenseNet201, (224, 224, 3), 1),
      (nets.MobileNet25, (224, 224, 3), 0),
      (nets.MobileNet50, (224, 224, 3), 1),
      (nets.MobileNet75, (224, 224, 3), 1),
      (nets.MobileNet100, (224, 224, 3), 0)])


test([(nets.ResNet50, (224, 224, 3), 0),
      (nets.ResNet101, (224, 224, 3), 0),
      (nets.ResNet152, (224, 224, 3), 0),
      (nets.ResNeXt50, (224, 224, 3), 0),
      (nets.ResNeXt101, (224, 224, 3), 1),
      (nets.ResNeXt101c64, (224, 224, 3), 1),
      (nets.WideResNet50, (224, 224, 3), 1)], 10)

test([(nets.ResNet50v2, (299, 299, 3), 0),
      (nets.ResNet101v2, (299, 299, 3), 1),
      (nets.ResNet152v2, (299, 299, 3), 1),
      (nets.ResNet200v2, (224, 224, 3), 0)], 10)

test([(nets.Inception1, (224, 224, 3), 0),
      (nets.Inception2, (224, 224, 3), 1),
      (nets.Inception3, (299, 299, 3), 0),
      (nets.Inception4, (299, 299, 3), 0),
      (nets.InceptionResNet2, (299, 299, 3), 1)], 10)

test([(nets.NASNetAlarge, (331, 331, 3), 0)], 10)

test([(nets.NASNetAmobile, (224, 224, 3), 0),
      (nets.VGG16, (224, 224, 3), 0),
      (nets.VGG19, (224, 224, 3), 1),
      (nets.SqueezeNet, (224, 224, 3), 1)], 10)

test([(nets.DenseNet121, (224, 224, 3), 0),
      (nets.DenseNet169, (224, 224, 3), 0),
      (nets.DenseNet201, (224, 224, 3), 1),
      (nets.MobileNet25, (224, 224, 3), 0),
      (nets.MobileNet50, (224, 224, 3), 1),
      (nets.MobileNet75, (224, 224, 3), 1),
      (nets.MobileNet100, (224, 224, 3), 0)], 10)
