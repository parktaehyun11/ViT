from tensorflow.keras.datasets import cifar10
from vit_trainer import VisualTransformer


# CIFAR10 데이터 다운로드
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images / 255.
test_images = test_images / 255.


hparam = {
    'learning_rate': 0.001,
    'num_classes': 10,
    'input_shape': (32, 32, 3),
    'weight_decay': 0.0001,
    'batch_size': 256,
    'num_epochs': 10,
    'image_size': 32,
    'obj_image_size': 21,
    'patch_size': 4,
    'projection_dim': 64,
    'num_heads': 8,
    'transformer_layers': 2,
}

ViT = VisualTransformer(**hparam)
ViT.training(train_images, train_labels, test_images, test_labels)
