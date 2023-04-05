import tensorflow as tf

# Define RBM model using TensorFlow 2
class RBM(tf.keras.Model):
    def __init__(self, _visible_bias, _hidden_units):
        super(RBM, self).__init__()
        self._visible_bias = tf.Variable(tf.random.normal([_visible_bias]), name='_visible_bias')
        self._hidden_units = tf.Variable(tf.random.normal([_hidden_units]), name='_hidden_bias')
        self._w_bias = tf.Variable(tf.random.normal([_visible_bias, _hidden_units]), name='_own_w')

    def call(self, inputs):
        hidden_layer = tf.nn.sigmoid(tf.matmul(inputs, self._w_bias) + self._hidden_units)
        reconstructed_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, tf.transpose(self._w_bias)) + self._visible_bias)
        return reconstructed_layer

    def sample_hidden(self, inputs):
        hidden_prob = tf.nn.sigmoid(tf.matmul(inputs, self._w_bias) + self._hidden_units)
        hidden_state = tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob))))
        return hidden_state

    def sample_visible(self, inputs):
        visible_prob = tf.nn.sigmoid(tf.matmul(inputs, self._w_bias, transpose_b=True) + self._visible_bias)
        visible_state = tf.nn.relu(tf.sign(visible_prob - tf.random.uniform(tf.shape(visible_prob))))
        return visible_state