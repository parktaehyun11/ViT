import tensorflow as tf
from tensorflow.keras.activations import gelu


class MultiHeadedAttention(tf.keras.Model):
    def __init__(self, dimension: int, heads: int = 8):
        super(MultiHeadedAttention, self).__init__()
        self.heads = heads
        self.dimension = dimension
        assert dimension // heads
        self.depth = dimension // heads
        self.wq = tf.keras.layers.Dense(dimension)
        self.wk = tf.keras.layers.Dense(dimension)
        self.wv = tf.keras.layers.Dense(dimension)
        self.dense = tf.keras.layers.Dense(dimension)

    def call(self, inputs):
        output = None
        batch_size = tf.shape(inputs)[0]
        q: tf.Tensor = self.wq(inputs)
        k: tf.Tensor = self.wk(inputs)
        v: tf.Tensor = self.wv(inputs)

        def split_heads(x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q, batch_size)
        k = split_heads(k, batch_size)
        v = split_heads(v, batch_size)

        def scaled_dot_product_attention(q, k, v):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            softmax = tf.nn.softmax(scaled_attention_logits, axis=-1)
            scaled_dot_product_attention_output = tf.matmul(softmax, v)
            return scaled_dot_product_attention_output, softmax

        attention_weights, softmax = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(attention_weights, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dimension))
        output = self.dense(concat_attention)
        return output


class ResidualBlock(tf.keras.Model):
    def __init__(self, residual_function):
        super(ResidualBlock, self).__init__()
        self.residual_function = residual_function

    def call(self, inputs):
        return self.residual_function(inputs) + inputs


class NormalizationBlock(tf.keras.Model):
    def __init__(self, norm_function, epsilon=1e-5):
        super(NormalizationBlock, self).__init__()
        self.norm_function = norm_function
        self.normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        return self.norm_function(self.normalize(inputs))


class MLPBlock(tf.keras.Model):
    def __init__(self, output_dimension, hidden_dimension):
        super(MLPBlock, self).__init__()
        self.output_dimension = tf.keras.layers.Dense(output_dimension)
        self.hidden_dimension = tf.keras.layers.Dense(hidden_dimension)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):
        output = None
        x = self.hidden_dimension(inputs)
        x = gelu(x)
        x = self.dropout1(x)
        x = self.output_dimension(x)
        x = gelu(x)
        output = self.dropout2(x)
        return output


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, dimension, depth, heads, mlp_dimension, image_size, patch_size):
        super(TransformerEncoder, self).__init__()
        layers_ = []
        layers_.append(tf.keras.Input(
            shape=((image_size // patch_size) * (image_size // patch_size) + 1, dimension)))
        for i in range(depth):
            layers_.append(NormalizationBlock(ResidualBlock(MultiHeadedAttention(dimension, heads))))
            layers_.append(NormalizationBlock(ResidualBlock(MLPBlock(dimension, mlp_dimension))))

        self.layers_ = tf.keras.Sequential(layers_)

    def call(self, inputs):
        return self.layers_(inputs)


class ImageTransformer(tf.keras.Model):
    def __init__(
            self, image_size, patch_size, n_classes, batch_size,
            dimension, depth, heads, mlp_dimension, channels=3):
        super(ImageTransformer, self).__init__()
        assert image_size % patch_size == 0, 'invalid patch size for image size'

        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dimension = dimension
        self.batch_size = batch_size

        self.positional_embedding = self.add_weight(
            "position_embeddings", shape=[num_patches + 1, dimension],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )
        self.classification_token = self.add_weight(
            "classification_token", shape=[1, 1, dimension],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )
        self.heads = heads
        self.depth = depth
        self.mlp_dimension = dimension
        self.n_classes = n_classes
        self.num_patches = num_patches
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_projection = tf.keras.layers.Dense(dimension)
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.MLP = MLPBlock(self.dimension, self.mlp_dimension)
        self.output_classes = tf.keras.layers.Dense(self.n_classes)
        self.transformer = TransformerEncoder(self.dimension, self.depth, self.heads, self.mlp_dimension,
                                              self.image_size, self.patch_size)
        self.dropout1 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs):
        output = None
        batch_size = tf.shape(inputs)[0]

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, patches.shape[1] * patches.shape[2], patch_dims])
        x = self.patch_projection(patches)

        cls_pos = tf.broadcast_to(
            self.classification_token, [batch_size, 1, self.dimension]
        )
        x = tf.concat([cls_pos, x], axis=1)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.normalization2(x)
        x = x[:, 0, :]
        x = self.dropout1(x)
        output = self.output_classes(x)
        return output
