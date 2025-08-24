_keras_base = None

try:
    # noinspection PyUnresolvedReferences
    import tensorflow.keras as _keras_base
    print("Using Keras from 'tensorflow.keras'")
except ImportError:
    # noinspection PyUnresolvedReferences
    import keras as _keras_base
    print("Using Keras from 'keras.src'")

keras_base = _keras_base
