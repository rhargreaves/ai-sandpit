import tensorflow as tf

# Check for available GPUs
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Force computation on the GPU
with tf.device("/GPU:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[2.0, 0.0], [1.0, 3.0]])
    c = tf.matmul(a, b)
    print("Result:", c)
