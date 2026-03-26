# src/utils.py

import os
import subprocess

# ====================================
# visualize the computational graph using graphviz
# ====================================
def _dot_var(v, verbose=False):
    '''Generate Graphviz representation of a Variable.

    Returns:
        string: v[label='name: data dtype', color=orange, style=filled]
    '''
    dot_var = '{}[label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.data) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    '''Generate Graphviz representation of a Function.

    Returns:
        string: f[label='class_name', color=lightblue, style=filled, shape=box]
                input: x -> f
                output: f -> y
    '''

    dot_func = '{}[label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # 使用 id() 来连接输入和输出的节点
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret

def get_dot_graph(output, verbose=True):
    '''
    Returns:
        string: Graphviz representation of the computational graph.
    '''
    ret = ''
    ret += _dot_var(output, verbose)
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    while funcs:
        func = funcs.pop()
        ret += _dot_func(func)
        for x in func.inputs:
            ret += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + ret + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    '''Plot the computational graph of a Variable.

    Returns:
        IPython.display.Image: Graph image
    '''
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    extension = os.path.splitext(to_file)[1][1:]                    # 拓展名
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    print(cmd)
    subprocess.run(cmd, shell=True)

    try:
        from IPython import display
        return display.Image(to_file)
    except:
        pass


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape: 返回值的 shape

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """sum backward 过程中 grad 形状扩张

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy