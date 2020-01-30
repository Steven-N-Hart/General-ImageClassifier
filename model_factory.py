import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras import Model

class GetModel:

    def __init__(self, model_name=None, img_size=256, classes=1, weights='imagenet', retrain=True, num_layers=None):
        self.model_name = model_name
        self.img_size = img_size
        self.classes = classes
        self.weights = weights
        self.num_layers = num_layers
        self.model, self.preprocess = self.__get_model_and_preprocess(retrain)

    def __get_model_and_preprocess(self, retrain):
        if retrain is True:
            include_top = False
        else:
            include_top = True

        input_tensor = Input(shape=(self.img_size, self.img_size, 3))
        weights = self.weights
        img_shape = (self.img_size, self.img_size, 3)

        if self.model_name == 'DenseNet121':
            model = tf.keras.applications.DenseNet121(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'DenseNet169':
            model = tf.keras.applications.DenseNet169(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'DenseNet201':
            model = tf.keras.applications.DenseNet201(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.densenet.preprocess_input(input_tensor)

        elif self.model_name == 'InceptionResNetV2':
            model = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=include_top,
                                                            input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input(input_tensor)

        elif self.model_name == 'InceptionV3':
            model = tf.keras.applications.InceptionV3(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.inception_v3.preprocess_input(input_tensor)

        elif self.model_name == 'MobileNet':
            model = tf.keras.applications.MobileNet(weights=weights, include_top=include_top,
                                                    input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.mobilenet.preprocess_input(input_tensor)

        elif self.model_name == 'MobileNetV2':
            model = tf.keras.applications.MobileNetV2(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(input_tensor)

        elif self.model_name == 'NASNetLarge':
            model = tf.keras.applications.NASNetLarge(weights=weights, include_top=include_top,
                                                      input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == 'NASNetMobile':
            model = tf.keras.applications.NASNetMobile(weights=weights, include_top=include_top,
                                                       input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.nasnet.preprocess_input(input_tensor)

        elif self.model_name == 'ResNet50':
            model = tf.keras.applications.ResNet50(weights=weights, include_top=include_top,
                                                   input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.resnet50.preprocess_input(input_tensor)

        elif self.model_name == 'VGG16':
            print('Model loaded was VGG16')
            model = tf.keras.applications.VGG16(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.vgg16.preprocess_input(input_tensor)

        elif self.model_name == 'VGG19':
            model = tf.keras.applications.VGG19(weights=weights, include_top=include_top,
                                                input_tensor=input_tensor, input_shape=img_shape)
            preprocess = tf.keras.applications.vgg19.preprocess_input(input_tensor)

        else:
            raise AttributeError("{} not found in available models".format(self.model_name))

        # Add a global average pooling and change the output size to our number of classes
        base_model = model
        base_model.trainable = False
        x = base_model.output
        x = Flatten()(x)
        out = Dense(self.classes, activation='softmax')(x)
        conv_model = Model(inputs=input_tensor, outputs=out)

        # Now check to see if we are retraining all but the head, or deeper down the stack
        if self.num_layers is not None:
            for layer in base_model.layers[:self.num_layers]:
                layer.trainable = False
            for layer in base_model.layers[self.num_layers:]:
                layer.trainable = True

        return conv_model, preprocess


    def _get_loss(self, name):
        if name == 'BinaryCrossentropy':
            return tf.keras.losses.BinaryCrossentropy()
        elif name == 'SparseCategoricalCrossentropy':
            print('Loss is SparseCategoricalCrossentropy')
            return tf.keras.losses.SparseCategoricalCrossentropy()
        elif name == 'CategoricalCrossentropy':
            return tf.keras.losses.CategoricalCrossentropy()
        else:
            raise AttributeError('{} as a loss function is not yet coded!'.format(name))

    def _get_optimizer(self, name, lr):

        if name == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
        elif name == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        elif name == 'Adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
        elif name == 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
        elif name == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif name == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise AttributeError("{} not found in available optimizers".format(self.model_name))
        return optimizer

    def compile_model(self, optimizer, lr, loss_name):
        model = self.model

        # Define the trainable model
        model.compile(optimizer=self._get_optimizer(optimizer, lr), loss=self._get_loss(loss_name),
                      metrics=[
                          #tf.keras.metrics.AUC(curve='PR', num_thresholds=10, name='PR'),
                          #tf.keras.metrics.AUC(num_thresholds=10, name='AUC'),
                          tf.keras.metrics.Accuracy(name='Accuracy'),
                          tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'),
                          #tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy')
                      ])

        return model
