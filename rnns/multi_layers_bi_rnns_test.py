import tensorflow as tf

from . import common_test_utils as common_utils

NUM_LAYERS = 4


class MultiLayerBiRNNsTest(tf.test.TestCase):

    def testMultiLayerBiLSTMs(self):
        fw_cells = common_utils.create_rnn_cells("lstm", common_utils.DEPTH, NUM_LAYERS)
        bw_cells = common_utils.create_rnn_cells("lstm", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_bi_rnns_results(
            fw_cells=fw_cells,
            bw_cells=bw_cells,
            inputs=inputs_placeholder,
            sequence_length=inputs_length_placeholder)
        # states is a tuple of (states_fw, states_bw) of length NUM_LAYERS
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is a tuple of ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h)) of length NUM_LAYERS
        self.assertEqual(2, len(states))
        states_fw = []
        states_bw = []
        for i in range(NUM_LAYERS):
            states_fw.append(states[0][i])
            states_bw.append(states[1][i])
        self.assertEqual(NUM_LAYERS, len(states_fw))
        self.assertEqual(NUM_LAYERS, len(states_bw))
        states_fw = tf.convert_to_tensor(states_fw)
        states_bw = tf.convert_to_tensor(states_bw)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            print("outputs shape: ", outputs.shape)
            print("states_fw shape: ", states_fw.shape)
            print("states_bw shape: ", states_bw.shape)
            # [batch_size, time_steps, depth*2]
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH * 2],
                                tf.shape(outputs))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_fw))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_bw))

    def testMultiLayerBiGRUs(self):
        fw_cells = common_utils.create_rnn_cells("gru", common_utils.DEPTH, NUM_LAYERS)
        bw_cells = common_utils.create_rnn_cells("gru", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_bi_rnns_results(
            fw_cells=fw_cells,
            bw_cells=bw_cells,
            inputs=inputs_placeholder,
            sequence_length=inputs_length_placeholder)
        self.assertEqual(2, len(states))
        # states is a tuple of (states_fw, states_bw) of length NUM_LAYERS
        states_fw = []
        states_bw = []
        for i in range(NUM_LAYERS):
            states_fw.append(states[0][i])
            states_bw.append(states[1][i])
        self.assertEqual(NUM_LAYERS, len(states_fw))
        self.assertEqual(NUM_LAYERS, len(states_bw))
        states_fw = tf.convert_to_tensor(states_fw)
        states_bw = tf.convert_to_tensor(states_bw)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })

            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH * 2],
                                tf.shape(outputs))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_fw))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_bw))

    def testMultiLayerBiNASs(self):
        fw_cells = common_utils.create_rnn_cells("nas", common_utils.DEPTH, NUM_LAYERS)
        bw_cells = common_utils.create_rnn_cells("nas", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_bi_rnns_results(
            fw_cells=fw_cells,
            bw_cells=bw_cells,
            inputs=inputs_placeholder,
            sequence_length=inputs_length_placeholder)
        # states is a tuple of (states_fw, states_bw) of length NUM_LAYERS
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is a tuple of ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h)) of length NUM_LAYERS
        self.assertEqual(2, len(states))
        states_fw = []
        states_bw = []
        for i in range(NUM_LAYERS):
            states_fw.append(states[0][i])
            states_bw.append(states[1][i])
        self.assertEqual(NUM_LAYERS, len(states_fw))
        self.assertEqual(NUM_LAYERS, len(states_bw))
        states_fw = tf.convert_to_tensor(states_fw)
        states_bw = tf.convert_to_tensor(states_bw)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            print("outputs shape: ", outputs.shape)
            print("states_fw shape: ", states_fw.shape)
            print("states_bw shape: ", states_bw.shape)
            # [batch_size, time_steps, depth*2]
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH * 2],
                                tf.shape(outputs))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_fw))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_bw))

    def testMultiLayerBiLayerNormLSTMs(self):
        fw_cells = common_utils.create_rnn_cells("layer_norm_lstm", common_utils.DEPTH, NUM_LAYERS)
        bw_cells = common_utils.create_rnn_cells("layer_norm_lstm", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_bi_rnns_results(
            fw_cells=fw_cells,
            bw_cells=bw_cells,
            inputs=inputs_placeholder,
            sequence_length=inputs_length_placeholder)
        # states is a tuple of (states_fw, states_bw) of length NUM_LAYERS
        # states_fw is a tuple of (states_fw_c, states_fw_h)
        # states_bw is a tuple of (states_bw_c, states_bw_h)
        # so states is a tuple of ((states_fw_c, states_fw_h), (states_bw_c, states_bw_h)) of length NUM_LAYERS
        self.assertEqual(2, len(states))
        states_fw = []
        states_bw = []
        for i in range(NUM_LAYERS):
            states_fw.append(states[0][i])
            states_bw.append(states[1][i])
        self.assertEqual(NUM_LAYERS, len(states_fw))
        self.assertEqual(NUM_LAYERS, len(states_bw))
        states_fw = tf.convert_to_tensor(states_fw)
        states_bw = tf.convert_to_tensor(states_bw)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states_fw, states_bw = sess.run(
                [outputs, states_fw, states_bw],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            print("outputs shape: ", outputs.shape)
            print("states_fw shape: ", states_fw.shape)
            print("states_bw shape: ", states_bw.shape)
            # [batch_size, time_steps, depth*2]
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH * 2],
                                tf.shape(outputs))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_fw))
            # [num_layers, 2, batch_size, depth]
            self.assertAllEqual([NUM_LAYERS, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_bw))


if __name__ == "__main__":
    tf.test.main()
