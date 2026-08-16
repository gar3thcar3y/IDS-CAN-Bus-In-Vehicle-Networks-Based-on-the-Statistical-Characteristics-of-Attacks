def manual_sinabs_to_brainscales2_nir(
    sinabs_model,
    output_file="CAN_IDS_BrainScaleS2.nir",
    input_shape=(1, 54, 8),
    max_dense_elements=50_000_000,
):
    """
    Convert a Sinabs/PyTorch SNN CNN into a BrainScaleS-2-friendly NIR graph.

    BrainScaleS-2 NIR path:
        Input
          -> Linear       (Conv2d lowered to a sparse matrix)
          -> LIF
          -> Linear       (AvgPool2d lowered to a sparse matrix)
          -> Linear       (Conv2d lowered to a sparse matrix)
          -> LIF
          -> Linear
          -> LIF
          -> Linear
          -> Output

    No nir.Conv2d, nir.AvgPool2d or nir.Flatten nodes are created.

    IMPORTANT
    ---------
    Lowering Conv2d to Linear preserves the convolution mathematically by
    constructing its equivalent sparse matrix. This can consume substantially
    more memory than a native convolution representation.

    The function assumes the same architecture as the original converter:

        Conv2d -> LIF -> AvgPool2d -> Conv2d -> LIF
        -> Flatten -> Linear -> LIF -> Linear

    The resulting NIR graph uses Linear + LIF/Affine primitives only.
    """

    import os
    import numpy as np
    import torch
    import nir

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return (
                value.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return np.asarray(value, dtype=np.float32)

    def tuple2(value):
        if isinstance(value, int):
            return (value, value)
        value = tuple(value)
        if len(value) == 1:
            return (value[0], value[0])
        if len(value) != 2:
            raise ValueError(f"Expected 2 values, got {value}")
        return (int(value[0]), int(value[1]))

    def get_attr(module, names, default=None):
        for name in names:
            if hasattr(module, name):
                value = getattr(module, name)
                if value is not None:
                    return value
        return default

    def expand_to_shape(value, target_shape, name):
        target_shape = tuple(int(x) for x in target_shape)
        value = to_numpy(value)

        if value is None:
            raise ValueError(f"{name} is None.")

        if value.ndim == 0 or value.size == 1:
            return np.full(
                target_shape,
                float(value.reshape(-1)[0]),
                dtype=np.float32,
            )

        if value.shape == target_shape:
            return value.astype(np.float32)

        if value.size == int(np.prod(target_shape)):
            return value.reshape(target_shape).astype(np.float32)

        raise ValueError(
            f"{name} has shape {value.shape}, "
            f"but expected {target_shape}."
        )

    def conv_output_shape(input_hw, kernel_size, stride, padding, dilation):
        h, w = input_hw
        kh, kw = tuple2(kernel_size)
        sh, sw = tuple2(stride)
        ph, pw = tuple2(padding)
        dh, dw = tuple2(dilation)

        out_h = ((h + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
        out_w = ((w + 2 * pw - dw * (kw - 1) - 1) // sw) + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid convolution output size: {(out_h, out_w)}"
            )

        return int(out_h), int(out_w)

    def pool_output_shape(input_hw, kernel_size, stride, padding):
        h, w = input_hw
        kh, kw = tuple2(kernel_size)
        sh, sw = tuple2(stride)
        ph, pw = tuple2(padding)

        out_h = ((h + 2 * ph - kh) // sh) + 1
        out_w = ((w + 2 * pw - kw) // sw) + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid pooling output size: {(out_h, out_w)}"
            )

        return int(out_h), int(out_w)

    # ------------------------------------------------------------
    # PyTorch module extraction
    # ------------------------------------------------------------

    if hasattr(sinabs_model, "spiking_model"):
        model = sinabs_model.spiking_model
    else:
        model = sinabs_model

    model = model.cpu()
    model.eval()

    direct_modules = list(model.children())

    print("=" * 72)
    print("SINABS -> BRAINSCales-2 FRIENDLY NIR")
    print("=" * 72)
    print("NIR version:", getattr(nir, "__version__", "unknown"))
    print("Input shape:", input_shape)

    print("\nDetected architecture:")
    for i, module in enumerate(direct_modules):
        print(f"  {i:2d}: {type(module).__name__}")

    conv_layers = [
        m for m in direct_modules
        if isinstance(m, torch.nn.Conv2d)
    ]

    pool_layers = [
        m for m in direct_modules
        if isinstance(m, torch.nn.AvgPool2d)
    ]

    linear_layers = [
        m for m in direct_modules
        if isinstance(m, torch.nn.Linear)
    ]

    spiking_layers = []
    for module in direct_modules:
        class_name = type(module).__name__.lower()
        if "lif" in class_name or "iaf" in class_name:
            spiking_layers.append(module)

    if len(conv_layers) != 2:
        raise RuntimeError(
            f"Expected exactly TWO Conv2d layers, found {len(conv_layers)}."
        )

    if len(pool_layers) != 1:
        raise RuntimeError(
            f"Expected exactly ONE AvgPool2d layer, found {len(pool_layers)}."
        )

    if len(linear_layers) != 2:
        raise RuntimeError(
            f"Expected exactly TWO Linear layers, found {len(linear_layers)}."
        )

    if len(spiking_layers) != 3:
        raise RuntimeError(
            f"Expected exactly THREE spiking layers, found {len(spiking_layers)}."
        )

    conv1, conv2 = conv_layers
    pool = pool_layers[0]
    linear1, linear2 = linear_layers
    lif1, lif2, lif3 = spiking_layers

    # ------------------------------------------------------------
    # Input / shape inference
    # ------------------------------------------------------------

    input_shape = tuple(int(x) for x in input_shape)

    if len(input_shape) != 3:
        raise ValueError(
            "input_shape must be (channels, height, width)."
        )

    in_channels = input_shape[0]
    input_hw = (input_shape[1], input_shape[2])

    if conv1.in_channels != in_channels:
        raise ValueError(
            f"Conv1 expects {conv1.in_channels} input channels, "
            f"but input_shape specifies {in_channels}."
        )

    if conv2.in_channels != conv1.out_channels:
        raise ValueError(
            "Conv2 input channels do not match Conv1 output channels."
        )

    conv1_hw = conv_output_shape(
        input_hw,
        conv1.kernel_size,
        conv1.stride,
        conv1.padding,
        conv1.dilation,
    )
    conv1_shape = (
        conv1.out_channels,
        conv1_hw[0],
        conv1_hw[1],
    )

    pool_hw = pool_output_shape(
        conv1_hw,
        pool.kernel_size,
        pool.stride,
        pool.padding,
    )
    pool_shape = (
        conv1.out_channels,
        pool_hw[0],
        pool_hw[1],
    )

    conv2_hw = conv_output_shape(
        pool_hw,
        conv2.kernel_size,
        conv2.stride,
        conv2.padding,
        conv2.dilation,
    )
    conv2_shape = (
        conv2.out_channels,
        conv2_hw[0],
        conv2_hw[1],
    )

    flatten_size = int(np.prod(conv2_shape))

    if linear1.in_features != flatten_size:
        raise ValueError(
            f"Linear1 expects {linear1.in_features} inputs, "
            f"but Conv2 produces {flatten_size} values."
        )

    if linear2.in_features != linear1.out_features:
        raise ValueError(
            "Linear2 input size does not match Linear1 output size."
        )

    print("\nCalculated shapes:")
    print("  Input       :", input_shape)
    print("  Conv1       :", conv1_shape)
    print("  AvgPool     :", pool_shape)
    print("  Conv2       :", conv2_shape)
    print("  Flatten     :", flatten_size)
    print("  Linear1     :", linear1.out_features)
    print("  Linear2     :", linear2.out_features)

    # ------------------------------------------------------------
    # Lower Conv2d -> sparse Linear
    # ------------------------------------------------------------

    def conv2d_to_sparse_matrix(layer, input_shape_chw, output_shape_chw):
        """
        Construct the exact matrix equivalent of a PyTorch Conv2d.

        PyTorch:
            y[oc, oh, ow] =
                sum(ic, kh, kw)
                W[oc, ic, kh, kw] * x[ic, ih, iw]
                + bias[oc]

        Matrix:
            y_flat = W_matrix @ x_flat + b_flat
        """

        in_c, in_h, in_w = input_shape_chw
        out_c, out_h, out_w = output_shape_chw

        kh, kw = tuple2(layer.kernel_size)
        sh, sw = tuple2(layer.stride)
        ph, pw = tuple2(layer.padding)
        dh, dw = tuple2(layer.dilation)

        groups = int(layer.groups)

        if groups != 1:
            raise NotImplementedError(
                "Grouped/depthwise Conv2d is not supported by this "
                "converter yet."
            )

        W = to_numpy(layer.weight)

        expected_shape = (
            out_c,
            in_c,
            kh,
            kw,
        )

        if W.shape != expected_shape:
            raise ValueError(
                f"Unexpected Conv2d weight shape {W.shape}; "
                f"expected {expected_shape}."
            )

        input_size = int(np.prod(input_shape_chw))
        output_size = int(np.prod(output_shape_chw))

        estimated_bytes = output_size * input_size * 4

        print("\nLowering Conv2d -> Linear")
        print("  Input neurons :", input_size)
        print("  Output neurons:", output_size)
        print(
            "  Dense-equivalent memory:",
            f"{estimated_bytes / (1024**2):.2f} MB",
        )

        if output_size * input_size > max_dense_elements:
            raise MemoryError(
                "The convolution-to-linear matrix would contain "
                f"{output_size * input_size:,} elements, exceeding "
                f"max_dense_elements={max_dense_elements:,}. "
                "Increase the limit only if you have enough RAM."
            )

        # NIR Linear expects a 2-D matrix.
        matrix = np.zeros(
            (output_size, input_size),
            dtype=np.float32,
        )

        bias = np.zeros(
            output_size,
            dtype=np.float32,
        )

        if layer.bias is not None:
            bias = np.asarray(
                to_numpy(layer.bias),
                dtype=np.float32,
            )

        # Flatten convention is C,H,W with W changing fastest.
        def flat_index(c, h, w, H, W):
            return (c * H + h) * W + w

        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):

                    out_idx = flat_index(
                        oc,
                        oh,
                        ow,
                        out_h,
                        out_w,
                    )

                    for ic in range(in_c):
                        for k_h in range(kh):
                            for k_w in range(kw):

                                ih = (
                                    oh * sh
                                    - ph
                                    + k_h * dh
                                )

                                iw = (
                                    ow * sw
                                    - pw
                                    + k_w * dw
                                )

                                # Padding contributes zero.
                                if (
                                    ih < 0
                                    or ih >= in_h
                                    or iw < 0
                                    or iw >= in_w
                                ):
                                    continue

                                in_idx = flat_index(
                                    ic,
                                    ih,
                                    iw,
                                    in_h,
                                    in_w,
                                )

                                matrix[out_idx, in_idx] = W[
                                    oc, ic, k_h, k_w
                                ]

        nonzero = np.count_nonzero(matrix)
        total = matrix.size

        print(
            f"  Non-zero weights: {nonzero:,}/{total:,} "
            f"({100.0 * nonzero / total:.4f}%)"
        )

        return matrix, bias

    # Conv1:
    W_conv1, B_conv1 = conv2d_to_sparse_matrix(
        conv1,
        input_shape,
        conv1_shape,
    )

    # Conv2:
    W_conv2, B_conv2 = conv2d_to_sparse_matrix(
        conv2,
        pool_shape,
        conv2_shape,
    )

    # ------------------------------------------------------------
    # Lower AvgPool2d -> Linear
    # ------------------------------------------------------------

    def avgpool_to_matrix(layer, input_shape_chw, output_shape_chw):
        in_c, in_h, in_w = input_shape_chw
        out_c, out_h, out_w = output_shape_chw

        if in_c != out_c:
            raise ValueError(
                "AvgPool channel count unexpectedly changed."
            )

        kh, kw = tuple2(layer.kernel_size)
        sh, sw = tuple2(layer.stride)
        ph, pw = tuple2(layer.padding)

        matrix = np.zeros(
            (
                int(np.prod(output_shape_chw)),
                int(np.prod(input_shape_chw)),
            ),
            dtype=np.float32,
        )

        def flat_index(c, h, w, H, W):
            return (c * H + h) * W + w

        # PyTorch AvgPool2d normally divides by kernel area for
        # non-overlapping valid windows. This implementation handles
        # the standard ceil_mode=False case used by the original model.
        scale = 1.0 / float(kh * kw)

        for c in range(in_c):
            for oh in range(out_h):
                for ow in range(out_w):

                    out_idx = flat_index(
                        c,
                        oh,
                        ow,
                        out_h,
                        out_w,
                    )

                    for k_h in range(kh):
                        for k_w in range(kw):

                            ih = oh * sh - ph + k_h
                            iw = ow * sw - pw + k_w

                            if (
                                ih < 0
                                or ih >= in_h
                                or iw < 0
                                or iw >= in_w
                            ):
                                continue

                            in_idx = flat_index(
                                c,
                                ih,
                                iw,
                                in_h,
                                in_w,
                            )

                            matrix[out_idx, in_idx] = scale

        print("\nLowering AvgPool2d -> Linear")
        print("  Input neurons :", matrix.shape[1])
        print("  Output neurons:", matrix.shape[0])

        return matrix

    W_pool = avgpool_to_matrix(
        pool,
        conv1_shape,
        pool_shape,
    )

    # ------------------------------------------------------------
    # LIF conversion
    # ------------------------------------------------------------

    def make_lif(layer, target_shape, name):
        tau_raw = get_attr(
            layer,
            ["tau_mem", "tau"],
        )

        threshold_raw = get_attr(
            layer,
            [
                "spike_threshold",
                "v_threshold",
                "threshold",
            ],
            1.0,
        )

        if tau_raw is None:
            raise RuntimeError(
                f"{name}: could not find tau_mem/tau."
            )

        tau = expand_to_shape(
            tau_raw,
            target_shape,
            f"{name}.tau",
        )

        threshold = expand_to_shape(
            threshold_raw,
            target_shape,
            f"{name}.v_threshold",
        )

        r = np.ones(
            target_shape,
            dtype=np.float32,
        )

        v_leak = np.zeros(
            target_shape,
            dtype=np.float32,
        )

        return nir.LIF(
            tau=tau,
            r=r,
            v_leak=v_leak,
            v_threshold=threshold,
        )

    # ------------------------------------------------------------
    # NIR graph
    # ------------------------------------------------------------

    nodes = {}
    edges = []

    nodes["input"] = nir.Input(
        input_type={
            "input": np.asarray(
                input_shape,
                dtype=np.int64,
            )
        }
    )

    # Conv1 -> Linear
    nodes["linear_conv1"] = nir.Linear(
        weight=W_conv1,
    )

    edges.append(
        ("input", "linear_conv1")
    )

    nodes["lif1"] = make_lif(
        lif1,
        conv1_shape,
        "lif1",
    )

    edges.append(
        ("linear_conv1", "lif1")
    )

    # AvgPool -> Linear
    nodes["linear_pool"] = nir.Linear(
        weight=W_pool,
    )

    edges.append(
        ("lif1", "linear_pool")
    )

    # Conv2 -> Linear
    nodes["linear_conv2"] = nir.Linear(
        weight=W_conv2,
    )

    edges.append(
        ("linear_pool", "linear_conv2")
    )

    nodes["lif2"] = make_lif(
        lif2,
        conv2_shape,
        "lif2",
    )

    edges.append(
        ("linear_conv2", "lif2")
    )

    # The explicit Flatten node is removed.
    # linear_conv2/lif2 output is already represented as a flat
    # vector of size np.prod(conv2_shape).

    W3 = to_numpy(linear1.weight)
    W4 = to_numpy(linear2.weight)

    B3 = (
        to_numpy(linear1.bias)
        if linear1.bias is not None
        else None
    )

    B4 = (
        to_numpy(linear2.bias)
        if linear2.bias is not None
        else None
    )

    # BrainScaleS-2-friendly Linear/Affine.
    # Keep bias exactly if present.
    if B3 is not None:
        nodes["linear1"] = nir.Affine(
            weight=W3,
            bias=B3,
        )
    else:
        nodes["linear1"] = nir.Linear(
            weight=W3,
        )

    edges.append(
        ("lif2", "linear1")
    )

    nodes["lif3"] = make_lif(
        lif3,
        (linear1.out_features,),
        "lif3",
    )

    edges.append(
        ("linear1", "lif3")
    )

    if B4 is not None:
        nodes["linear2"] = nir.Affine(
            weight=W4,
            bias=B4,
        )
    else:
        nodes["linear2"] = nir.Linear(
            weight=W4,
        )

    edges.append(
        ("lif3", "linear2")
    )

    nodes["output"] = nir.Output(
        output_type={
            "output": np.asarray(
                [linear2.out_features],
                dtype=np.int64,
            )
        }
    )

    edges.append(
        ("linear2", "output")
    )

    # ------------------------------------------------------------
    # Create graph
    # ------------------------------------------------------------

    print("\nCreating NIR graph...")

    graph = nir.NIRGraph(
        nodes=nodes,
        edges=edges,
    )

    print("Running NIR type inference...")
    graph.infer_types()
    print("✓ NIR type inference succeeded.")

    # ------------------------------------------------------------
    # Strict BrainScaleS-2 compatibility check
    # ------------------------------------------------------------

    allowed = {
        "Input",
        "Output",
        "Linear",
        "Affine",
        "LIF",
    }

    print("\nBrainScaleS-2 compatibility check:")

    incompatible = []

    for name, node in graph.nodes.items():
        node_type = type(node).__name__

        print(
            f"  {name:16s} -> {node_type}"
        )

        if node_type not in allowed:
            incompatible.append(
                (name, node_type)
            )

    if incompatible:
        raise RuntimeError(
            "Incompatible NIR nodes remain: "
            + str(incompatible)
        )

    print(
        "✓ No Conv2d, AvgPool2d or Flatten nodes remain."
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print(
        f"\nWriting BrainScaleS-2 NIR:\n"
        f"{output_file}"
    )

    nir.write(
        output_file,
        graph,
    )

    if not os.path.exists(output_file):
        raise RuntimeError(
            "NIR file was not created."
        )

    # ------------------------------------------------------------
    # Reload test
    # ------------------------------------------------------------

    print("\nReloading saved NIR...")
    loaded_graph = nir.read(output_file)

    loaded_graph.infer_types()

    print("✓ NIR file successfully reloaded.")

    print("\nFinal graph:")
    for name, node in loaded_graph.nodes.items():
        print(
            f"  {name:16s} -> "
            f"{type(node).__name__}"
        )

    print("\nEdges:")
    for source, target in loaded_graph.edges:
        print(
            f"  {source:16s} -> {target}"
        )

    # Final safety check after serialization.
    bad_after_reload = [
        (name, type(node).__name__)
        for name, node in loaded_graph.nodes.items()
        if type(node).__name__ not in allowed
    ]

    if bad_after_reload:
        raise RuntimeError(
            "Incompatible nodes appeared after reload: "
            + str(bad_after_reload)
        )

    print("\n" + "=" * 72)
    print("BRAINScales-2 NIR EXPORT SUCCESSFUL")
    print("=" * 72)
    print(
        f"\nSaved to:\n"
        f"{os.path.abspath(output_file)}"
    )

    return loaded_graph
