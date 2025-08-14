_keras_base = None

try:
    import tensorflow.keras as _keras_base
    print("Using Keras from 'tensorflow.keras'")
except ImportError:
    import keras as _keras_base
    print("Using Keras from 'keras.src'")


keras_base = _keras_base
