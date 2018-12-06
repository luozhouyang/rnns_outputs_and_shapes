import tensorflow as tf

from . import common_test_utils as common_utils


class SingleLayerBiRNNsTest(tf.test.TestCase):

    def testSingleLayerBiLSTM(self):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(num_units=common_utils.DEPTH)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_bi_rnn_results(
            cell_fw=lstm_fw,
            cell_bw=lstm_bw,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states is a tuple of (states_fw, states_bw)
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is: ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h))
        states_fw, states_bw = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            states_fw_c, states_fw_h = states_fw
            states_bw_c, states_bw_h = states_bw
            print("single layer bi LSTM outputs :\n%s\n\n" % outputs)
            print("single layer bi LSTM states_fw_c:\n%s\n\n" % states_fw_c)
            print("single layer bi LSTM states_fw_h:\n%s\n\n" % states_fw_h)
            print("single layer bi LSTM states_bw_c:\n%s\n\n" % states_bw_c)
            print("single layer bi LSTM states_bw_h:\n%s\n\n" % states_bw_h)
            # [8,5,3*2]
            print("    outputs shape of single layer bi LSTM: ", outputs.shape)
            # [8,3]
            print("states_fw_c shape of single layer bi LSTM: ", states_fw_c.shape)
            # [8,3]
            print("states_fw_h shape of single layer bi LSTM: ", states_fw_h.shape)
            # [8,3]
            print("states_bw_c shape of single layer bi LSTM: ", states_bw_c.shape)
            # [8,3]
            print("states_bw_h shape of single layer bi LSTM: ", states_bw_h.shape)

    def testSingleLayerBiGRU(self):
        gru_fw = tf.nn.rnn_cell.GRUCell(num_units=common_utils.DEPTH)
        gru_bw = tf.nn.rnn_cell.GRUCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_bi_rnn_results(
            cell_fw=gru_fw,
            cell_bw=gru_bw,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states is a tuple of (states_fw, states_bw)
        # unlike lstm, which has two outputs(c, h), gru cell has only one output
        states_fw, states_bw = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            print("single layer bi LSTM outputs :\n%s\n\n" % outputs)
            print("single layer bi GRU states_fw: \n%s\n\n" % states_fw)
            print("single layer bi GRU states_bw: \n%s\n\n" % states_bw)
            # [8,5,3*2]
            print("  outputs shape of single layer bi GRU: ", outputs.shape)
            # [8,3]
            print("states_fw shape of single layer bi GRU: ", states_fw.shape)
            # [8,3]
            print("states_fw shape of single layer bi GRU: ", states_bw.shape)

    def testSingleLayerBiNAS(self):
        nas_fw = tf.contrib.rnn.NASCell(num_units=common_utils.DEPTH)
        nas_bw = tf.contrib.rnn.NASCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_bi_rnn_results(
            cell_fw=nas_fw,
            cell_bw=nas_bw,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states is a tuple of (states_fw, states_bw)
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is: ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h))
        states_fw, states_bw = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            states_fw_c, states_fw_h = states_fw
            states_bw_c, states_bw_h = states_bw
            print("single layer bi NAS outputs :\n%s\n\n" % outputs)
            print("single layer bi NAS states_fw_c:\n%s\n\n" % states_fw_c)
            print("single layer bi NAS states_fw_h:\n%s\n\n" % states_fw_h)
            print("single layer bi NAS states_bw_c:\n%s\n\n" % states_bw_c)
            print("single layer bi NAS states_bw_h:\n%s\n\n" % states_bw_h)
            # [8,5,3*2]
            print("    outputs shape of single layer bi NAS: ", outputs.shape)
            # [8,3]
            print("states_fw_c shape of single layer bi NAS: ", states_fw_c.shape)
            # [8,3]
            print("states_fw_h shape of single layer bi NAS: ", states_fw_h.shape)
            # [8,3]
            print("states_bw_c shape of single layer bi NAS: ", states_bw_c.shape)
            # [8,3]
            print("states_bw_h shape of single layer bi NAS: ", states_bw_h.shape)

    def testSingleLayerBiLayerNormLSTM(self):
        lstm_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=common_utils.DEPTH)
        lstm_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=common_utils.DEPTH)
        seq_placeholder = tf.placeholder(dtype=tf.float32.as_numpy_dtype,
                                         shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        seq_len_placeholder = tf.placeholder(dtype=tf.int32.as_numpy_dtype, shape=(None))
        outputs, states = common_utils.get_bi_rnn_results(
            cell_fw=lstm_fw,
            cell_bw=lstm_bw,
            inputs=seq_placeholder,
            sequence_length=seq_len_placeholder)
        # states is a tuple of (states_fw, states_bw)
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is: ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h))
        states_fw, states_bw = states
        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    seq_placeholder: inputs,
                    seq_len_placeholder: inputs_length
                })
            states_fw_c, states_fw_h = states_fw
            states_bw_c, states_bw_h = states_bw
            print("single layer bi LSTM outputs :\n%s\n\n" % outputs)
            print("single layer bi LSTM states_fw_c:\n%s\n\n" % states_fw_c)
            print("single layer bi LSTM states_fw_h:\n%s\n\n" % states_fw_h)
            print("single layer bi LSTM states_bw_c:\n%s\n\n" % states_bw_c)
            print("single layer bi LSTM states_bw_h:\n%s\n\n" % states_bw_h)
            # [8,5,3*2]
            print("    outputs shape of single layer bi LayerNormLSTM: ", outputs.shape)
            # [8,3]
            print("states_fw_c shape of single layer bi LayerNormLSTM: ", states_fw_c.shape)
            # [8,3]
            print("states_fw_h shape of single layer bi LayerNormLSTM: ", states_fw_h.shape)
            # [8,3]
            print("states_bw_c shape of single layer bi LayerNormLSTM: ", states_bw_c.shape)
            # [8,3]
            print("states_bw_h shape of single layer bi LayerNormLSTM: ", states_bw_h.shape)


if __name__ == "__main__":
    tf.test.main()
