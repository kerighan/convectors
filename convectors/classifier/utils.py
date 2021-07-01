def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    import os
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper
        deprecation.deprecated = deprecated

    except ImportError:
        pass
