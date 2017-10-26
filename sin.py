from math import sin, pi
import keras

INPUT_SIZE = 10
DOMAIN_MAX = 2 * pi

sin_input = [i * DOMAIN_MAX / INPUT_SIZE for i in range(INPUT_SIZE)]
sin_output = [sin(x) for x in sin_input]

model = keras.models.Sequential(layers=[
    keras.layers.Dense(units=2, activation=keras.activations.sigmoid, input_dim=1, ),
    keras.layers.Dense(units=1, activation=keras.activations.sigmoid),
])

model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mean_squared_error)

model.fit(sin_input, sin_output, epochs=10)

