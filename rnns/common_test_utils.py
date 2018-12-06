import numpy as np
import tensorflow as tf

TIME_STEPS = 5
DEPTH = 3
BATCH_SIZE = 8


def get_batched_inputs():
    data = np.array([
        [[0, 1, 2], [1, 2, 3], [3, 4, 5], [1, 2, 3], [3, 4, 5]],
        [[2, 3, 4], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # pad with 0
        [[5, 3, 2], [7, 8, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # pad with 0
        [[0, 2, 1], [6, 7, 8], [6, 9, 0], [6, 7, 8], [6, 9, 0]],
        [[0, 1, 2], [1, 2, 3], [3, 4, 5], [1, 2, 3], [3, 4, 5]],
        [[2, 3, 4], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # pad with 0
        [[5, 3, 2], [7, 8, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # pad with 0
        [[0, 2, 1], [6, 7, 8], [6, 9, 0], [6, 7, 8], [6, 9, 0]]
    ], dtype=tf.float32.as_numpy_dtype)
    seq_len = np.array([5, 1, 2, 5, 5, 1, 2, 5], dtype=tf.int32.as_numpy_dtype)
    return data, seq_len


def get_uni_rnn_results(uni_cell, inputs, sequence_length):
    outputs, states = tf.nn.dynamic_rnn(
        cell=uni_cell,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)
    return outputs, states


def get_bi_rnn_results(cell_fw, cell_bw, inputs, sequence_length):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)
    outputs = tf.concat(outputs, -1)
    return outputs, states


def get_uni_rnns_results(uni_cells, inputs, sequence_length):
    outputs, states = tf.nn.dynamic_rnn(
        cell=uni_cells,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)
    return outputs, states


def get_bi_rnns_results(fw_cells, bw_cells, inputs, sequence_length):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cells,
        cell_bw=bw_cells,
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)
    outputs = tf.concat(outputs, -1)
    return outputs, states


def create_rnn_cell(name, num_units):
    if name == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell(num_units)
    elif name == "gru":
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    elif name == "nas":
        cell = tf.contrib.rnn.NASCell(num_units)
    elif name == "layer_norm_lstm":
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
    else:
        raise ValueError("Invalid name %s" % name)
    return cell


def create_rnn_cells(name, num_units, num_layers):
    cells = []
    for _ in range(num_layers):
        cells.append(create_rnn_cell(name, num_units))
    return tf.nn.rnn_cell.MultiRNNCell(cells)
