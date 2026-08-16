def manual_sinabs_to_nir(
    sinabs_model,
    output_file="CAN_IDS_SNN_manual.nir",
    input_shape=(1, 54, 8),
):
    """
    Manually convert a Sinabs SNN / PyTorch sequential model
    into NIR 1.0.4.

    No sinabs.to_nir() is used.

    The model is NEVER executed during conversion.

    Supported architecture:

        Input
          ↓
        Conv2d
          ↓
        LIF
          ↓
        AvgPool2d
          ↓
        Conv2d
          ↓
        LIF
          ↓
        Flatten
          ↓
        Linear / Affine
          ↓
        LIF
          ↓
        Linear / Affine
          ↓
        Output

    Designed for NIR 1.0.4.
    """

    import os
    import inspect
    import numpy as np
    import torch
    import nir

    # ============================================================
    # Helpers
    # ============================================================

    def to_numpy(value):
        """Convert Torch/Numpy/scalar to float32 NumPy."""

        if value is None:
            return None

        if torch.is_tensor(value):
            return (
                value.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return np.asarray(
            value,
            dtype=np.float32
        )


    def tuple2(value):
        """Convert int / tuple / torch.Size to a 2-tuple."""

        if isinstance(value, int):
            return (value, value)

        value = tuple(value)

        if len(value) == 1:
            return (value[0], value[0])

        if len(value) != 2:
            raise ValueError(
                f"Expected 2 values, got {value}"
            )

        return (
            int(value[0]),
            int(value[1])
        )


    def get_attr(
        module,
        names,
        default=None
    ):
        """Get first existing attribute."""

        for name in names:

            if hasattr(module, name):

                value = getattr(
                    module,
                    name
                )

                if value is not None:
                    return value

        return default


    def expand_to_shape(
        value,
        target_shape,
        name
    ):
        """
        Convert a scalar / 1D / already-shaped LIF parameter
        to the exact NIR neuron tensor shape.

        IMPORTANT:
        This controls the parameter shape.

        It is NOT the same thing as flattening the layer.
        """

        target_shape = tuple(
            int(x)
            for x in target_shape
        )

        value = to_numpy(value)

        if value is None:

            raise ValueError(
                f"{name} is None."
            )

        # --------------------------------------------------------
        # Scalar
        # --------------------------------------------------------

        if value.ndim == 0:

            return np.full(
                target_shape,
                float(value),
                dtype=np.float32
            )

        # --------------------------------------------------------
        # One value
        # --------------------------------------------------------

        if value.size == 1:

            return np.full(
                target_shape,
                float(value.reshape(-1)[0]),
                dtype=np.float32
            )

        # --------------------------------------------------------
        # Already correct shape
        # --------------------------------------------------------

        if value.shape == target_shape:

            return value.astype(
                np.float32
            )

        # --------------------------------------------------------
        # Same number of elements
        # --------------------------------------------------------

        if value.size == int(
            np.prod(target_shape)
        ):

            return value.reshape(
                target_shape
            ).astype(
                np.float32
            )

        raise ValueError(
            f"{name} has shape {value.shape}, "
            f"but expected {target_shape}."
        )


    def conv_output_shape(
        input_hw,
        kernel_size,
        stride,
        padding,
        dilation
    ):
        """
        PyTorch Conv2d output shape.
        """

        h, w = input_hw

        kh, kw = tuple2(
            kernel_size
        )

        sh, sw = tuple2(
            stride
        )

        ph, pw = tuple2(
            padding
        )

        dh, dw = tuple2(
            dilation
        )

        out_h = (
            (
                h
                + 2 * ph
                - dh * (kh - 1)
                - 1
            )
            // sh
        ) + 1

        out_w = (
            (
                w
                + 2 * pw
                - dw * (kw - 1)
                - 1
            )
            // sw
        ) + 1

        return (
            int(out_h),
            int(out_w)
        )


    def pool_output_shape(
        input_hw,
        kernel_size,
        stride,
        padding
    ):
        """
        PyTorch AvgPool2d output shape for
        ceil_mode=False.
        """

        h, w = input_hw

        kh, kw = tuple2(
            kernel_size
        )

        sh, sw = tuple2(
            stride
        )

        ph, pw = tuple2(
            padding
        )

        out_h = (
            (
                h
                + 2 * ph
                - kh
            )
            // sh
        ) + 1

        out_w = (
            (
                w
                + 2 * pw
                - kw
            )
            // sw
        ) + 1

        return (
            int(out_h),
            int(out_w)
        )


    # ============================================================
    # Locate actual Sinabs model
    # ============================================================

    if hasattr(
        sinabs_model,
        "spiking_model"
    ):

        model = (
            sinabs_model
            .spiking_model
        )

    else:

        model = sinabs_model


    model = model.cpu()
    model.eval()


    # ============================================================
    # Print versions
    # ============================================================

    print("=" * 72)
    print("MANUAL SINABS → NIR")
    print("=" * 72)

    print(
        "NIR version:",
        getattr(
            nir,
            "__version__",
            "unknown"
        )
    )

    print(
        "Input shape:",
        input_shape
    )


    # ============================================================
    # Collect modules
    # ============================================================

    modules = [
        module
        for module in model.modules()
        if module is not model
    ]


    # Only direct Sequential children are normally wanted.
    # If the model is a Sequential this gives the architecture
    # in exactly the same order as the original network.

    if hasattr(
        model,
        "children"
    ):

        direct_modules = list(
            model.children()
        )

    else:

        direct_modules = modules


    print("\nDetected architecture:")

    for i, module in enumerate(
        direct_modules
    ):

        print(
            f"  {i:2d}: "
            f"{type(module).__name__}"
        )


    # ============================================================
    # Find important layers
    # ============================================================

    conv_layers = [
        m
        for m in direct_modules
        if isinstance(
            m,
            torch.nn.Conv2d
        )
    ]


    pool_layers = [
        m
        for m in direct_modules
        if isinstance(
            m,
            torch.nn.AvgPool2d
        )
    ]


    linear_layers = [
        m
        for m in direct_modules
        if isinstance(
            m,
            torch.nn.Linear
        )
    ]


    spiking_layers = []

    for module in direct_modules:

        class_name = (
            type(module)
            .__name__
            .lower()
        )

        if (
            "lif" in class_name
            or "iaf" in class_name
        ):

            spiking_layers.append(
                module
            )


    print(
        "\nConv2d layers:",
        len(conv_layers)
    )

    print(
        "AvgPool2d layers:",
        len(pool_layers)
    )

    print(
        "Linear layers:",
        len(linear_layers)
    )

    print(
        "Spiking layers:",
        len(spiking_layers)
    )


    # ============================================================
    # Validate expected architecture
    # ============================================================

    if len(conv_layers) != 2:

        raise RuntimeError(
            "Expected exactly TWO Conv2d "
            f"layers, found {len(conv_layers)}."
        )


    if len(pool_layers) != 1:

        raise RuntimeError(
            "Expected exactly ONE AvgPool2d "
            f"layer, found {len(pool_layers)}."
        )


    if len(linear_layers) != 2:

        raise RuntimeError(
            "Expected exactly TWO Linear "
            f"layers, found {len(linear_layers)}."
        )


    if len(spiking_layers) != 3:

        raise RuntimeError(
            "Expected exactly THREE spiking "
            f"layers, found {len(spiking_layers)}."
        )


    # ============================================================
    # Assign layers
    # ============================================================

    conv1 = conv_layers[0]
    conv2 = conv_layers[1]

    pool = pool_layers[0]

    linear1 = linear_layers[0]
    linear2 = linear_layers[1]

    lif1 = spiking_layers[0]
    lif2 = spiking_layers[1]
    lif3 = spiking_layers[2]


    # ============================================================
    # Input
    # ============================================================

    input_shape = tuple(
        int(x)
        for x in input_shape
    )

    if len(input_shape) != 3:

        raise ValueError(
            "input_shape must be "
            "(channels, height, width)."
        )


    in_channels = input_shape[0]
    input_hw = (
        input_shape[1],
        input_shape[2]
    )


    # ============================================================
    # Conv1 output
    # ============================================================

    conv1_hw = conv_output_shape(
        input_hw,

        conv1.kernel_size,

        conv1.stride,

        conv1.padding,

        conv1.dilation
    )


    conv1_shape = (
        conv1.out_channels,
        conv1_hw[0],
        conv1_hw[1]
    )


    # ============================================================
    # Pool output
    # ============================================================

    pool_hw = pool_output_shape(
        conv1_hw,

        pool.kernel_size,

        pool.stride,

        pool.padding
    )


    pool_shape = (
        conv1.out_channels,
        pool_hw[0],
        pool_hw[1]
    )


    # ============================================================
    # Conv2 output
    # ============================================================

    conv2_hw = conv_output_shape(
        pool_hw,

        conv2.kernel_size,

        conv2.stride,

        conv2.padding,

        conv2.dilation
    )


    conv2_shape = (
        conv2.out_channels,
        conv2_hw[0],
        conv2_hw[1]
    )


    # ============================================================
    # Flatten
    # ============================================================

    flatten_size = int(
        np.prod(
            conv2_shape
        )
    )


    # ============================================================
    # Print shapes
    # ============================================================

    print("\nCalculated tensor shapes:")

    print(
        "  Input      :",
        input_shape
    )

    print(
        "  Conv1      :",
        conv1_shape
    )

    print(
        "  LIF1       :",
        conv1_shape
    )

    print(
        "  AvgPool    :",
        pool_shape
    )

    print(
        "  Conv2      :",
        conv2_shape
    )

    print(
        "  LIF2       :",
        conv2_shape
    )

    print(
        "  Flatten    :",
        flatten_size
    )

    print(
        "  Linear1    :",
        linear1.out_features
    )

    print(
        "  LIF3       :",
        linear1.out_features
    )

    print(
        "  Linear2    :",
        linear2.out_features
    )


    # ============================================================
    # Validate channels
    # ============================================================

    if conv1.in_channels != in_channels:

        raise ValueError(
            f"Conv1 expects "
            f"{conv1.in_channels} input channels, "
            f"but input_shape specifies "
            f"{in_channels}."
        )


    if conv2.in_channels != conv1.out_channels:

        raise ValueError(
            "Conv2 input channels do not "
            "match Conv1 output channels."
        )


    if linear1.in_features != flatten_size:

        raise ValueError(
            f"Linear1 expects "
            f"{linear1.in_features} inputs, "
            f"but Flatten produces "
            f"{flatten_size}."
        )


    if linear2.in_features != linear1.out_features:

        raise ValueError(
            "Linear2 input size does not "
            "match Linear1 output size."
        )


    # ============================================================
    # Extract weights
    # ============================================================

    W1 = to_numpy(
        conv1.weight
    )

    W2 = to_numpy(
        conv2.weight
    )

    W3 = to_numpy(
        linear1.weight
    )

    W4 = to_numpy(
        linear2.weight
    )


    # ============================================================
    # Extract biases
    # ============================================================

    B1 = (
        to_numpy(
            conv1.bias
        )
        if conv1.bias is not None
        else np.zeros(
            conv1.out_channels,
            dtype=np.float32
        )
    )


    B2 = (
        to_numpy(
            conv2.bias
        )
        if conv2.bias is not None
        else np.zeros(
            conv2.out_channels,
            dtype=np.float32
        )
    )


    B3 = (
        to_numpy(
            linear1.bias
        )
        if linear1.bias is not None
        else np.zeros(
            linear1.out_features,
            dtype=np.float32
        )
    )


    B4 = (
        to_numpy(
            linear2.bias
        )
        if linear2.bias is not None
        else np.zeros(
            linear2.out_features,
            dtype=np.float32
        )
    )


    # ============================================================
    # Extract LIF parameters
    # ============================================================

    def make_lif(
        layer,
        target_shape,
        name
    ):

        tau_raw = get_attr(
            layer,
            [
                "tau_mem",
                "tau"
            ]
        )


        threshold_raw = get_attr(
            layer,
            [
                "spike_threshold",
                "v_threshold",
                "threshold"
            ],
            1.0
        )


        if tau_raw is None:

            raise RuntimeError(
                f"{name}: could not find "
                "tau_mem/tau."
            )


        tau = expand_to_shape(
            tau_raw,
            target_shape,
            f"{name}.tau"
        )


        threshold = expand_to_shape(
            threshold_raw,
            target_shape,
            f"{name}.v_threshold"
        )


        # Sinabs LIF uses zero leak and unit resistance
        # in the standard NIR mapping.
        #
        # This is also how Sinabs' own NIR conversion
        # represents LIF.

        r = np.ones(
            target_shape,
            dtype=np.float32
        )


        v_leak = np.zeros(
            target_shape,
            dtype=np.float32
        )


        return nir.LIF(

            tau=tau,

            r=r,

            v_leak=v_leak,

            v_threshold=threshold
        )


    # ============================================================
    # Create nodes
    # ============================================================

    nodes = {}
    edges = []


    # ============================================================
    # Input
    # ============================================================

    nodes["input"] = nir.Input(

        input_type={
            "input": np.asarray(
                input_shape,
                dtype=np.int64
            )
        }
    )


    # ============================================================
    # Conv1
    # ============================================================

    nodes["conv1"] = nir.Conv2d(

        input_shape=(
            input_shape[1],
            input_shape[2]
        ),

        weight=W1,

        stride=tuple2(
            conv1.stride
        ),

        padding=tuple2(
            conv1.padding
        ),

        dilation=tuple2(
            conv1.dilation
        ),

        groups=int(
            conv1.groups
        ),

        bias=B1
    )


    edges.append(
        (
            "input",
            "conv1"
        )
    )


    # ============================================================
    # LIF1
    # ============================================================

    nodes["lif1"] = make_lif(
        lif1,
        conv1_shape,
        "lif1"
    )


    edges.append(
        (
            "conv1",
            "lif1"
        )
    )


    # ============================================================
    # AvgPool2d
    # ============================================================

    #
    # NIR 1.0.4 provides AvgPool2d.
    #
    # The official NIR PyTorch mapping uses:
    #
    # nir.AvgPool2d(
    #     kernel_size=...,
    #     stride=...,
    #     padding=...
    # )
    #

    nodes["avgpool"] = nir.AvgPool2d(

        kernel_size=np.asarray(
            tuple2(
                pool.kernel_size
            ),
            dtype=np.int64
        ),

        stride=np.asarray(
            tuple2(
                pool.stride
            ),
            dtype=np.int64
        ),

        padding=np.asarray(
            tuple2(
                pool.padding
            ),
            dtype=np.int64
        )
    )


    edges.append(
        (
            "lif1",
            "avgpool"
        )
    )


    # ============================================================
    # Conv2
    # ============================================================

    nodes["conv2"] = nir.Conv2d(

        input_shape=(
            pool_shape[1],
            pool_shape[2]
        ),

        weight=W2,

        stride=tuple2(
            conv2.stride
        ),

        padding=tuple2(
            conv2.padding
        ),

        dilation=tuple2(
            conv2.dilation
        ),

        groups=int(
            conv2.groups
        ),

        bias=B2
    )


    edges.append(
        (
            "avgpool",
            "conv2"
        )
    )


    # ============================================================
    # LIF2
    # ============================================================

    nodes["lif2"] = make_lif(
        lif2,
        conv2_shape,
        "lif2"
    )


    edges.append(
        (
            "conv2",
            "lif2"
        )
    )


    # ============================================================
    # Flatten
    # ============================================================

    nodes["flatten"] = nir.Flatten(

        input_type=np.asarray(
            conv2_shape,
            dtype=np.int64
        ),

        start_dim=0,

        end_dim=2
    )


    edges.append(
        (
            "lif2",
            "flatten"
        )
    )


    # ============================================================
    # Linear 1
    # ============================================================
    #
    # NIR Linear has no bias in your NIR 1.0.4 API.
    #
    # Therefore use Affine for PyTorch Linear layers
    # that have bias.
    #
    # Affine implements:
    #
    #     y = W x + b
    #
    # This preserves the original PyTorch model.
    # ============================================================

    if linear1.bias is not None:

        nodes["linear1"] = nir.Affine(

            weight=W3,

            bias=B3
        )

    else:

        nodes["linear1"] = nir.Linear(
            weight=W3
        )


    edges.append(
        (
            "flatten",
            "linear1"
        )
    )


    # ============================================================
    # LIF3
    # ============================================================

    nodes["lif3"] = make_lif(
        lif3,
        (linear1.out_features,),
        "lif3"
    )


    edges.append(
        (
            "linear1",
            "lif3"
        )
    )


    # ============================================================
    # Linear 2
    # ============================================================

    if linear2.bias is not None:

        nodes["linear2"] = nir.Affine(

            weight=W4,

            bias=B4
        )

    else:

        nodes["linear2"] = nir.Linear(
            weight=W4
        )


    edges.append(
        (
            "lif3",
            "linear2"
        )
    )


    # ============================================================
    # Output
    # ============================================================

    nodes["output"] = nir.Output(

        output_type={
            "output": np.asarray(
                [linear2.out_features],
                dtype=np.int64
            )
        }
    )


    edges.append(
        (
            "linear2",
            "output"
        )
    )


    # ============================================================
    # Create NIR graph
    # ============================================================

    print(
        "\nCreating NIR graph..."
    )


    graph = nir.NIRGraph(
        nodes=nodes,
        edges=edges
    )


    # ============================================================
    # Type inference
    # ============================================================

    print(
        "Running NIR type inference..."
    )


    graph.infer_types()


    print(
        "✓ NIR type inference succeeded."
    )


    # ============================================================
    # Display inferred interfaces
    # ============================================================

    print(
        "\nNIR interfaces:"
    )


    for name, node in graph.nodes.items():

        print(
            f"\n{name}"
        )

        print(
            "  type:",
            type(node).__name__
        )

        print(
            "  input:",
            getattr(
                node,
                "input_type",
                None
            )
        )

        print(
            "  output:",
            getattr(
                node,
                "output_type",
                None
            )
        )


    # ============================================================
    # Save
    # ============================================================

    print(
        f"\nWriting:\n{output_file}"
    )


    nir.write(
        output_file,
        graph
    )


    if not os.path.exists(
        output_file
    ):

        raise RuntimeError(
            "NIR file was not created."
        )


    print(
        "✓ NIR file written."
    )


    # ============================================================
    # Reload test
    # ============================================================

    print(
        "\nReloading saved NIR file..."
    )


    loaded_graph = nir.read(
        output_file
    )


    print(
        "✓ NIR file successfully reloaded."
    )


    # ============================================================
    # Print final graph
    # ============================================================

    print(
        "\nFinal NIR graph:"
    )


    for name, node in (
        loaded_graph.nodes.items()
    ):

        print(
            f"  {name:10s} "
            f"→ {type(node).__name__}"
        )


    print(
        "\nEdges:"
    )


    for source, target in (
        loaded_graph.edges
    ):

        print(
            f"  {source:10s} "
            f"→ {target}"
        )


    # ============================================================
    # Final success
    # ============================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "NIR EXPORT SUCCESSFUL"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSaved to:\n{os.path.abspath(output_file)}"
    )


    return loaded_graph