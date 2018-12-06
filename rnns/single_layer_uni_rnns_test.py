import tensorflow as tf

from . import common_test_utils as common_utils


class SingleLayerUniRNNsTest(tf.test.TestCase):

    def testSingleLayerUniLSTM(self):
        uni_lstm = tf.nn.rnn_cell.LSTMCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_uni_rnn_results(
            uni_cell=uni_lstm,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states if a tuple of (states_c, states_h)
        states_c, states_h = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_c, states_h = sess.run(
                [outputs, states_c, states_h],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            print("single layer uni LSTM outputs :\n%s\n\n" % outputs)
            print("single layer uni LSTM states_c:\n%s\n\n" % states_c)
            print("single layer uni LSTM states_h:\n%s\n\n" % states_h)
            # [8,5,3]
            print(" outputs shape of single layer uni LSTM: ", outputs.shape)
            # [8,3]
            print("states_c shape of single layer uni LSTM: ", states_c.shape)
            # [8,3]
            print("states_h shape of single layer uni LSTM: ", states_h.shape)

    def testSingleLayerUniGRU(self):
        uni_gru = tf.nn.rnn_cell.GRUCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_uni_rnn_results(
            uni_cell=uni_gru,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # unlike lstm, which has two outputs(c, h), gru has only one output
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states = sess.run(
                [outputs, states],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            print("single layer uni GRU outputs :\n%s\n\n" % outputs)
            print("single layer uni GRU states_c:\n%s\n\n" % states)
            # [8,5,3]
            print("outputs shape of single layer uni GRU: ", outputs.shape)
            # [8,3]
            print(" states shape of single layer uni GRU: ", states.shape)

    def testSingleLayerUniNAS(self):
        uni_nas = tf.contrib.rnn.NASCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_uni_rnn_results(
            uni_cell=uni_nas,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states if a tuple of (states_c, states_h)
        states_c, states_h = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_c, states_h = sess.run(
                [outputs, states_c, states_h],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            print("single layer uni NAS outputs :\n%s\n\n" % outputs)
            print("single layer uni NAS states_c:\n%s\n\n" % states_c)
            print("single layer uni NAS states_h:\n%s\n\n" % states_h)
            # [8,5,3]
            print(" outputs shape of single layer uni NAS: ", outputs.shape)
            # [8,3]
            print("states_c shape of single layer uni NAS: ", states_c.shape)
            # [8,3]
            print("states_h shape of single layer uni NAS: ", states_h.shape)

    def testSingleLayerUniLayerNormLSTM(self):
        uni_layer_norm_lstm = tf.nn.rnn_cell.LSTMCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_uni_rnn_results(
            uni_cell=uni_layer_norm_lstm,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states is a tuple of (states_c, states_h)
        states_c, states_h = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_c, states_h = sess.run(
                [outputs, states_c, states_h],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            print("single layer uni LayerNormLSTM outputs :\n%s\n\n" % outputs)
            print("single layer uni LayerNormLSTM states_c:\n%s\n\n" % states_c)
            print("single layer uni LayerNormLSTM states_h:\n%s\n\n" % states_h)
            # [8,5,3]
            print(" outputs shape of single layer uni LayerNormLSTM: ", outputs.shape)
            # [8,3]
            print("states_c shape of single layer uni LayerNormLSTM: ", states_c.shape)
            # [8,3]
            print("states_h shape of single layer uni LayerNormLSTM: ", states_h.shape)


if __name__ == "__main__":
    tf.test.main()
