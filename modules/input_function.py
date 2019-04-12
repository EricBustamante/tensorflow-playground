import tensorflow as tf

def my_input_fn(filenames, batch_size=1, shuffle=True, num_epochs=None):  

    ds = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            "label": tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
            "image": tf.FixedLenFeature((), tf.string, default_value="")
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.reshape(image, [80, 28, 1])
        label = tf.cast(parsed["label"], tf.int32)
        return image, label

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(100000)
    #ds = ds.batch(batch_size)
    #ds = ds.map(map_func=parser, num_parallel_calls = 16)
    ds = ds.apply(tf.contrib.data.map_and_batch(
        map_func=parser, batch_size=batch_size))
    ds = ds.repeat(num_epochs)
    ds = ds.prefetch(buffer_size=batch_size)
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return {"x": features}, labels