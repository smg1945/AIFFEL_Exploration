import tensorflow as tf

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation_rate=1, act=True):
        super(Conv2D, self).__init__()
        self.act = act
        self.conv = tf.keras.layers.Conv2D(out_c, kernel_size, padding=padding, dilation_rate=dilation_rate, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()

        self.fc1 = tf.keras.layers.Conv2D(in_planes // ratio, 1, use_bias=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Conv2D(in_planes, 1, use_bias=False)

        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(tf.expand_dims(self.avg_pool(x), axis=-1))))
        max_out = self.fc2(self.relu1(self.fc1(tf.expand_dims(self.max_pool(x), axis=-1))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = tf.keras.layers.Conv2D(1, kernel_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_out = tf.expand_dims(tf.reduce_mean(x, axis=-1), axis=-1)
        max_out = tf.expand_dims(tf.reduce_max(x, axis=-1), axis=-1)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv1(x)
        return x * self.sigmoid(x)

class DilatedConv(tf.keras.layers.Layer):
    def __init__(self, in_c, out_c):
        super(DilatedConv, self).__init__()
        self.relu = tf.keras.layers.ReLU()

        self.c1 = tf.keras.Sequential([Conv2D(in_c, out_c, kernel_size=1, padding='same'), ChannelAttention(out_c)])
        self.c2 = tf.keras.Sequential([Conv2D(in_c, out_c, kernel_size=3, padding='same', dilation_rate=6), ChannelAttention(out_c)])
        self.c3 = tf.keras.Sequential([Conv2D(in_c, out_c, kernel_size=3, padding='same', dilation_rate=12), ChannelAttention(out_c)])
        self.c4 = tf.keras.Sequential([Conv2D(in_c, out_c, kernel_size=3, padding='same', dilation_rate=18), ChannelAttention(out_c)])
        self.c5 = Conv2D(out_c*4, out_c, kernel_size=3, padding='same', act=False)
        self.c6 = Conv2D(in_c, out_c, kernel_size=1, padding='same', act=False)
        self.sa = SpatialAttention()

    def call(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = tf.concat([x1, x2, x3, x4], axis=-1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc+xs)
        x = self.sa(x)
        return x

class LabelAttention(tf.keras.models.Model):
    def __init__(self, in_c):
        super(LabelAttention, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.c1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=in_c[0], kernel_size=1, padding='valid', use_bias=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=in_c[0], kernel_size=1, padding='valid', use_bias=False),
        ])

    def call(self, feats, label):
        b, c = label.shape
        label = tf.reshape(label, (b, c, 1, 1))
        ch_attn = self.c1(label)
        ch_map = tf.sigmoid(ch_attn)
        feats = feats * ch_map
        ch_attn = tf.reshape(ch_attn, (ch_attn.shape[0], ch_attn.shape[1]))
        return ch_attn, feats


class DecoderBlock(tf.keras.models.Model):
    def __init__(self, in_c, out_c, scale=2):
        super(DecoderBlock, self).__init__()
        self.scale = scale
        self.up = tf.keras.layers.UpSampling2D(size=(scale, scale))
        self.c1 = tf.keras.layers.Conv2D(out_c, kernel_size=1, padding='valid')
        self.c2 = tf.keras.layers.Conv2D(out_c, kernel_size=3, padding='same')
        self.c3 = tf.keras.layers.Conv2D(out_c, kernel_size=3, padding='same')
        self.c4 = tf.keras.layers.Conv2D(out_c, kernel_size=1, padding='valid')
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()
        self.add = tf.keras.layers.Add()

    def call(self, x, skip):
        x = self.up(x)
        x = tf.concat([x, skip], axis=3)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = tf.keras.activations.relu(self.add([x, s1]))

        s2 = x
        x = self.c3(x)
        x = tf.keras.activations.relu(self.add([x, s2, s1]))

        s3 = x
        x = self.c4(x)
        x = tf.keras.activations.relu(self.add([x, s3, s2, s1]))

        x = self.ca(x)
        x = self.sa(x)
        return x

class OutputBlock(tf.keras.models.Model):
    def __init__(self, in_c, out_c):
        super(OutputBlock, self).__init__()
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.c1 = tf.keras.layers.Conv2D(out_c, kernel_size=1, padding='valid')

    def call(self, x):
        x = self.up(x)
        x = self.c1(x)
        return x

class TextClassifier(tf.keras.models.Model):
    def __init__(self, in_c, out_c):
        super(TextClassifier, self).__init__()
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(in_c, in_c))
        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Dense(in_c//8, use_bias=False), tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(out_c[0], use_bias=False)
        ])
        self.fc2 = tf.keras.Sequential([
            tf.keras.layers.Dense(in_c//8, use_bias=False), tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(out_c[1], use_bias=False)
        ])

    def call(self, feats):
        pool = self.avg_pool(feats)
        pool = tf.keras.layers.Flatten()(pool)
        num_polyps = self.fc1(pool)
        polyp_sizes = self.fc2(pool)
        return num_polyps, polyp_sizes

class EmbeddingFeatureFusion(tf.keras.models.Model):
    def __init__(self, in_c, out_c):
        super(EmbeddingFeatureFusion, self).__init__()
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_c, kernel_size=1, use_bias=False), tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(out_c, kernel_size=1, use_bias=False), tf.keras.layers.ReLU()
        ])

    def call(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = tf.nn.softmax(num_polyps, axis=1)
        polyp_sizes_prob = tf.nn.softmax(polyp_sizes, axis=1)
        prob = tf.concat([num_polyps_prob, polyp_sizes_prob], axis=1)
        prob = tf.keras.layers.Reshape((prob.shape[0], prob.shape[1], 1))(prob)
        x = label * prob
        x = tf.keras.layers.Reshape((x.shape[0], -1, 1, 1))(x)
        x = self.fc(x)
        x = tf.keras.layers.Reshape((x.shape[0], -1))(x)
        return x

class Conv2DRelu(tf.keras.layers.Layer):
    def __init__(self, out_c, kernel_size=1, padding='valid', act=True):
        super(Conv2DRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_c, kernel_size=kernel_size, padding=padding)
        self.act = act

    def call(self, inputs):
        x = self.conv(inputs)
        if self.act:
            x = tf.nn.relu(x)
        return x

class MultiscaleFeatureAggregation(tf.keras.layers.Layer):
    def __init__(self, in_c, out_c):
        super(MultiscaleFeatureAggregation, self).__init__()

        self.up_2x2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.up_4x4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')

        self.c11 = Conv2DRelu(in_c[0], out_c)
        self.c12 = Conv2DRelu(in_c[1], out_c)
        self.c13 = Conv2DRelu(in_c[2], out_c)
        self.c14 = Conv2DRelu(out_c * 3, out_c)

        self.c2 = Conv2DRelu(out_c, out_c, act=False)
        self.c3 = Conv2DRelu(out_c, out_c, act=False)

    def call(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)

        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = tf.nn.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = tf.nn.relu(x + s2 + s1)

        return x

class TGAPolypSeg(tf.keras.Model):
    def __init__(self):
        super(TGAPolypSeg, self).__init__()

        """ Backbone: ResNet50 """
        backbone = tf.keras.applications.ResNet50(include_top=False, weights=None)

        self.layer0 = tf.keras.Sequential(backbone.layers[:4])
        self.layer1 = tf.keras.Sequential(backbone.layers[4:6])
        self.layer2 = backbone.get_layer('conv2_block3_out')
        self.layer3 = backbone.get_layer('conv3_block4_out')
        self.layer4 = backbone.get_layer('conv4_block6_out')
        
        self.text_classifier = TextClassifier(1024, [2, 3])
        self.label_fc = EmbeddingFeatureFusion([2, 3, 300], 128)

        """ Dilated Conv """
        self.s1 = DilatedConv(64, 128)
        self.s2 = DilatedConv(256, 128)
        self.s3 = DilatedConv(512, 128)
        self.s4 = DilatedConv(1024, 128)

        """ Decoder """
        self.d1 = DecoderBlock(128, 128, scale=2)
        self.a1 = LabelAttention([128, 128])

        self.d2 = DecoderBlock(128, 128, scale=2)
        self.a2 = LabelAttention([128, 128])

        self.d3 = DecoderBlock(128, 128, scale=2)
        self.a3 = LabelAttention([128, 128])

        self.ag = MultiscaleFeatureAggregation([128, 128, 128], 128)

        self.y1 = OutputBlock(128, 1)

    def call(self, inputs):
        image, label = inputs
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)    ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)    ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)    ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)    ## [-1, 1024, h/16, w/16]
        x5 = self.layer4(x4)    ## [-1, 2048, h/32, w/32]

        num_polyps, polyp_sizes = self.text_classifier(x5)
        f0 = self.label_fc([num_polyps, polyp_sizes, label])

        """ Dilated Conv """
        s1 = self.s1(x1)
        s2 = self.s2(x2)
        s3 = self.s3(x3)
        s4 = self.s4(x4)
        s5 = self.s5(x5)

        """ Decoder """
        d1 = self.d1([s5, s4])
        f1, a1 = self.a1([d1, f0])

        d2 = self.d2([a1, s3])
        f = f0 + f1
        f2, a2 = self.a2([d2, f])

        d3 = self.d3([a2, s2])
        f = f + f2
        f3, a3 = self.a3([d3, f])

        ag = self.ag([a1, a2, a3])
        y1 = self.y1(ag)

        return y1, num_polyps, polyp_sizes

def get_flops(model):
    # convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        [tf.TensorSpec([1, 3, 256, 256], tf.float32), tf.TensorSpec([1, 5, 300], tf.float32)])

    # get frozen ConcreteFunction
    frozen_func = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(concrete_function)
    frozen_func.graph.as_graph_def()

    with tf.python.profiler.profile('graph.pbtxt', options=tf.python.profiler.ProfilerOptions(host_tracer_level=2, python_tracer_level=1, device_tracer_level=1)) as _:
        tf.profiler.experimental.start('logdir')
        result = frozen_func(tf.keras.layers.Input('input:0', shape=(1, 3, 256, 256), dtype=tf.float32), tf.keras.layers.Input('input:0', shape=(1, 5, 300), dtype=tf.float32))
        tf.profiler.experimental.stop()
    
    return result

if __name__ == "__main__":
    model = TGAPolypSeg()
    result = get_flops(model)

    print('      - Flops:  ' + str(result['flops']))
    print('      - Params: ' + str(result['params']))