import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 80, 28, 1])
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size,  80, 28, 1]
  # Output Tensor Shape: [batch_size, 80, 28, 35]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=42,
      kernel_size=[8, 8],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 80, 28, 35]
  # Output Tensor Shape: [batch_size, 40, 14, 35]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 40, 14, 35]
  # Output Tensor Shape: [batch_size, 40, 14, 70]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=84,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 40, 14, 70]
  # Output Tensor Shape: [batch_size, 20, 7, 70]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

   # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 40, 14, 35]
  # Output Tensor Shape: [batch_size, 40, 14, 70]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=126,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 40, 14, 70]
  # Output Tensor Shape: [batch_size, 20, 7, 70]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 15, 5, 128]
  # Output Tensor Shape: [batch_size, 15 * 5 * 128]
  pool3_flat = tf.reshape(pool3, [-1, 10*3*126])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 50 * 17 * 70]
  # Output Tensor Shape: [batch_size, 1020]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=2040, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=1020, activation=tf.nn.relu)

  # Add dropout operation; 0.4 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
  reshaped = tf.reshape(dense2, [-1, 6, 170])
  # Logits layer
  # Input Tensor Shape: [batch_size, 6, 170]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(
      inputs=reshaped, 
      units=26)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=2, name="argmax_tensor"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    #gradients, variables = [list(t) for t in zip(*optimizer.compute_gradients(loss))]
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    #train_op = optimizer.apply_gradients(zip(gradients, variables), 
    #                                     global_step=tf.train.get_global_step())
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
