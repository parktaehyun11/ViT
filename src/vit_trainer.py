from vit_models import ImageTransformer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow_addons.optimizers import AdamW

class VisualTransformer:
    def __init__(self, **hparam):
        self.learning_rate = hparam['learning_rate']
        self.num_classes = hparam['num_classes']
        self.input_shape = hparam['input_shape']
        self.weight_decay = hparam['weight_decay']
        self.batch_size = hparam['batch_size']
        self.num_epochs = hparam['num_epochs']
        self.image_size = hparam['image_size']
        self.obj_image_size = hparam['obj_image_size']
        self.patch_size = hparam['patch_size']
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = hparam['projection_dim']
        self.num_heads = hparam['num_heads']
        self.transformer_layers = hparam['transformer_layers']

        # build model
        self.set_vit_model()

        # set compile
        self.set_optimizer()
        self.set_loss()
        self.set_metrics()

        self.set_compile()

    def set_vit_model(self):
        self.vit_model = ImageTransformer(self.image_size, self.patch_size, self.num_classes, self.batch_size,
                                          self.projection_dim, self.transformer_layers, self.num_heads,
                                          self.projection_dim)

    def set_optimizer(self):
        self.opt = AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

    def set_loss(self):
        self.loss = SparseCategoricalCrossentropy(from_logits=True)

    def set_metrics(self):
        self.metrics = SparseCategoricalAccuracy(name="accuracy")

    def set_compile(self):
        self.vit_model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

    def training(self, train_x, train_y, test_x, test_y):
        for i in range(self.num_epochs):
            print('Step : {}\n'.format(i + 1))
            hist = self.vit_model.fit(
                x=train_x,
                y=train_y,
                batch_size=self.batch_size,
                epochs=1,
                validation_split=0.1,
                shuffle=True
            )

            _, accuracy = self.vit_model.evaluate(test_x, test_y)
            print('Test Accuracy :', accuracy)
