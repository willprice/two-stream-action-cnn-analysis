def get_layer_output_shapes(net):
    return _get_layer_shapes(net, net.top_names)


def get_layer_input_shapes(net):
    return _get_layer_shapes(net, net.bottom_names)

def filter_shapes(net):
    filter_shapes = dict()
    for layer, params in net.params.items():
        weights = params[0]
        biases = params[1]
        filter_shapes[layer] = tuple(weights.data.shape)

    return filter_shapes

def _get_layer_shapes(net, blob_names):
    layer_shapes = dict()
    for layer_name in blob_names.keys():
        input_blob_names = blob_names[layer_name]
        input_blobs = [net.blobs[name] for name in input_blob_names]
        input_shapes = [tuple(input_blob.shape) for input_blob in input_blobs]
        layer_shapes[layer_name] = input_shapes
    return layer_shapes


def print_layer_shapes(net, direction='output'):
    if direction == "output":
        layer_shapes = get_layer_output_shapes(net)
    elif direction == "input":
        layer_shapes = get_layer_input_shapes(net)
    else:
        raise RuntimeError("Expected 'input' or 'output' for `direction` arg.")
    print("Layer shapes ({})\n".format(direction))
    for layer in net.top_names.keys():
        print("{!s:<10}: {!s:<10}".format(layer, layer_shapes[layer]))