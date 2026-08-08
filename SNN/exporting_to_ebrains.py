from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(54, 8, 1)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dense(
        2,
        activation="softmax"
    )
])

conv1 = model.layers[0]
conv2 = model.layers[1]
dense1 = model.layers[3]
dense2 = model.layers[4]

W_conv1, b_conv1 = conv1.get_weights()
W_conv2, b_conv2 = conv2.get_weights()
W_dense1, b_dense1 = dense1.get_weights()
W_dense2, b_dense2 = dense2.get_weights()

np.savez_compressed(
        weights_file,
        W_conv1=W_conv1,
        b_conv1=b_conv1,
        W_conv2=W_conv2,
        b_conv2=b_conv2,
        W_dense1=W_dense1,
        b_dense1=b_dense1,
        W_dense2=W_dense2,
        b_dense2=b_dense2,
    )

print(W_conv1.shape)
print(b_conv1.shape)

print(W_conv2.shape)
print(b_conv2.shape)

print(W_dense1.shape)
print(b_dense1.shape)

print(W_dense2.shape)
print(b_dense2.shape)

import pyNN.spiNNaker as sim

sim.setup(timestep=1.0)

input_population = sim.Population(
    54 * 8 * 1,
    sim.SpikeSourceArray(
        spike_times=spike_times
    ),
    label="input"
)

conv1_population = sim.Population(
    32 * 52 * 6,
    sim.IF_curr_exp(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_thresh=-50.0,
        tau_syn_E=5.0,
        tau_syn_I=5.0
    ),
    label="conv1"
)

def create_conv2d_connections(
    weights,
    input_height,
    input_width,
    input_channels,
    output_channels,
    kernel_height,
    kernel_width,
    stride=1,
    padding=0
):
    connections = []

    output_height = (
        input_height + 2 * padding - kernel_height
    ) // stride + 1

    output_width = (
        input_width + 2 * padding - kernel_width
    ) // stride + 1

    for out_y in range(output_height):
        for out_x in range(output_width):
            for out_c in range(output_channels):

                post_index = (
                    out_c * output_height * output_width
                    + out_y * output_width
                    + out_x
                )

                for ky in range(kernel_height):
                    for kx in range(kernel_width):
                        for in_c in range(input_channels):

                            in_y = (
                                out_y * stride
                                + ky
                                - padding
                            )

                            in_x = (
                                out_x * stride
                                + kx
                                - padding
                            )

                            # Ignore positions outside
                            # the image
                            if (
                                in_y < 0
                                or in_y >= input_height
                                or in_x < 0
                                or in_x >= input_width
                            ):
                                continue

                            pre_index = (
                                in_c * input_height * input_width
                                + in_y * input_width
                                + in_x
                            )

                            weight = weights[
                                ky,
                                kx,
                                in_c,
                                out_c
                            ]

                            # Don't create zero-weight synapses
                            if weight != 0:

                                connections.append(
                                    (
                                        pre_index,
                                        post_index,
                                        float(weight),
                                        1.0
                                    )
                                )

    return connections


conv1_connections = create_conv2d_connections(
    W_conv1,
    input_height=54,
    input_width=8,
    input_channels=1,
    output_channels=32,
    kernel_height=3,
    kernel_width=3
)

conv1_projection = sim.Projection(
    input_population,
    conv1_population,
    sim.FromListConnector(conv1_connections)
)

conv2_population = sim.Population(
    64 * 50 * 4,
    sim.IF_curr_exp(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_thresh=-50.0,
        tau_syn_E=5.0,
        tau_syn_I=5.0
    ),
    label="conv2"
)

conv2_connections = create_conv2d_connections(
    W_conv2,
    input_height=52,
    input_width=6,
    input_channels=32,
    output_channels=64,
    kernel_height=3,
    kernel_width=3
)

conv2_projection = sim.Projection(
    conv1_population,
    conv2_population,
    sim.FromListConnector(conv2_connections)
)

dense1_population = sim.Population(
    64,
    sim.IF_curr_exp(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_thresh=-50.0,
        tau_syn_E=5.0,
        tau_syn_I=5.0
    ),
    label="dense1"
)

dense1_connections = []

for pre in range(12800):
    for post in range(64):

        weight = W_dense1[pre, post]

        if weight != 0:

            dense1_connections.append(
                (
                    pre,
                    post,
                    float(weight),
                    1.0
                )
            )

dense1_projection = sim.Projection(
    conv2_population,
    dense1_population,
    sim.FromListConnector(dense1_connections)
)

output_population = sim.Population(
    2,
    sim.IF_curr_exp(
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-65.0,
        v_thresh=-50.0,
        tau_syn_E=5.0,
        tau_syn_I=5.0
    ),
    label="output"
)

dense2_connections = []

for pre in range(64):
    for post in range(2):

        weight = W_dense2[pre, post]

        if weight != 0:

            dense2_connections.append(
                (
                    pre,
                    post,
                    float(weight),
                    1.0
                )
            )

dense2_projection = sim.Projection(
    dense1_population,
    output_population,
    sim.FromListConnector(dense2_connections)
)