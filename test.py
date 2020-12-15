import tensorflow as tf

if (__name__ == "__main__"):
  print(f"Is cuda activated: {tf.test.is_built_with_cuda()}")
  print(tf.config.list_physical_devices('GPU')) 