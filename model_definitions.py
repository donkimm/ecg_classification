# Contains functions to build models.
import tensorflow as tf


def intermediate_block(input_layer):
    """
    Define the intermediate block that gets used throughout the model
    """
    x_skip = tf.keras.layers.MaxPooling1D(pool_size=1, strides=1)(input_layer)
    
    x = tf.keras.layers.BatchNormalization()(input_layer)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(x)
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(x)
    
    x = tf.keras.layers.Add()([x, x_skip])
    
    return x
    
    
def stanford_resnet(shape, classes, blocks):
    """
    Define the Stanford Arrythmia Resnet for Heartbeat Classification
    https://stanfordmlgroup.github.io/projects/ecg/
    """
    
    # First Block
    inputs = tf.keras.layers.Input(shape)
    
    x = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # First Skip Connection
    x_skip = tf.keras.layers.MaxPooling1D(pool_size=1, strides=1)(x)
    
    # Second Block
    x = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='same')(x)
    
    # Merge Skip Connection
    x = tf.keras.layers.Add()([x, x_skip])

    # Intermediate Layers
    for i in range(blocks):
        x = intermediate_block(x)
        
    # Ending Block
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    # Finish Model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Currently using Accuracy for the metric, but might be worth considering a different metric 
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
