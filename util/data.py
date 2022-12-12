import tensorflow as tf
from util.image import load_image_train, load_image_test

class Dataset:
    
    def __init__(self,
                train_path,
                x_file_pattern,
                y_file_pattern,
                test_path,
                test_x_file_pattern = "",
                test_y_file_pattern = "",
                different_pattern=False,
                buffer_size = 4000,
                batch_size = 1,
                seed = 42):
                        
        if different_pattern == False:
            test_x_file_pattern = x_file_pattern
            test_y_file_pattern = y_file_pattern
        
        
        train_x_ds = tf.data.Dataset.list_files(
            train_path+x_file_pattern, seed=seed)

        train_y_ds = tf.data.Dataset.list_files(
            train_path+y_file_pattern, seed=seed)
        
        train_ds = tf.data.Dataset.zip((train_x_ds,train_y_ds))
        train_ds = train_ds.map(load_image_train,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.shuffle(buffer_size)
        self._train_ds = train_ds.batch(batch_size)
        
        test_x_ds = tf.data.Dataset.list_files(
            test_path+test_x_file_pattern, seed=seed
        )
        test_y_ds = tf.data.Dataset.list_files(
            test_path+test_y_file_pattern, seed=seed
        )
        test_ds = tf.data.Dataset.zip((test_x_ds, test_y_ds))
        test_ds = test_ds.map(load_image_test)
        self._test_ds = test_ds.batch(batch_size)
        
    @property
    def train_ds(self):
        return self._train_ds
    
    @property
    def test_ds(self):
        return self._test_ds    