import tensorflow as tf

from . import common_test_utils as common_utils

NUM_LAYERS = 4


class MultiLayersUniRNNsTest(tf.test.TestCase):

    def testMultiLayerUniLSTMs(self):
        cells = common_utils.create_rnn_cells("lstm", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_uni_rnns_results(
            cells,
            inputs_placeholder,
            inputs_length_placeholder)
        # states if a tuple of (states_c, states_h) of length NUM_LAYERS
        states_c = []
        states_h = []
        for i in range(NUM_LAYERS):
            states_c.append(states[i][0])
            states_h.append(states[i][1])
        states_c = tf.convert_to_tensor(states_c)
        states_h = tf.convert_to_tensor(states_h)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states, states_c, states_h = sess.run(
                [outputs, states, states_c, states_h],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            self.assertEqual(NUM_LAYERS, len(states))
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH],
                                tf.shape(outputs))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_c))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_h))

    def testMultiLayerUniGRUs(self):
        cells = common_utils.create_rnn_cells("gru", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_uni_rnns_results(
            cells,
            inputs_placeholder,
            inputs_length_placeholder)
        # states is a tuple of (state) of length NUM_LAYERS
        self.assertEqual(NUM_LAYERS, len(states))
        states_list = []
        for i in range(len(states)):
            states_list.append(states[i])
        states = tf.convert_to_tensor(states_list)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states = sess.run(
                [outputs, states],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH],
                                tf.shape(outputs))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states))

    def testMultiLayerUniNASs(self):
        cells = common_utils.create_rnn_cells("nas", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_uni_rnns_results(
            cells,
            inputs_placeholder,
            inputs_length_placeholder)
        # states if a tuple of (states_c, states_h) of length NUM_LAYERS
        states_c = []
        states_h = []
        for i in range(NUM_LAYERS):
            states_c.append(states[i][0])
            states_h.append(states[i][1])
        states_c = tf.convert_to_tensor(states_c)
        states_h = tf.convert_to_tensor(states_h)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states, states_c, states_h = sess.run(
                [outputs, states, states_c, states_h],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            self.assertEqual(NUM_LAYERS, len(states))
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH],
                                tf.shape(outputs))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_c))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_h))

    def testMultiLayerUniLayerNormLSTM(self):
        cells = common_utils.create_rnn_cells("layer_norm_lstm", common_utils.DEPTH, NUM_LAYERS)
        inputs_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
        inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=(None))
        outputs, states = common_utils.get_uni_rnns_results(
            cells,
            inputs_placeholder,
            inputs_length_placeholder)
        # states if a tuple of (states_c, states_h) of length NUM_LAYERS
        states_c = []
        states_h = []
        for i in range(NUM_LAYERS):
            states_c.append(states[i][0])
            states_h.append(states[i][1])
        states_c = tf.convert_to_tensor(states_c)
        states_h = tf.convert_to_tensor(states_h)

        inputs, inputs_length = common_utils.get_batched_inputs()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, states, states_c, states_h = sess.run(
                [outputs, states, states_c, states_h],
                feed_dict={
                    inputs_placeholder: inputs,
                    inputs_length_placeholder: inputs_length
                })
            self.assertEqual(NUM_LAYERS, len(states))
            self.assertAllEqual([common_utils.BATCH_SIZE, common_utils.TIME_STEPS, common_utils.DEPTH],
                                tf.shape(outputs))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_c))
            self.assertAllEqual([NUM_LAYERS, common_utils.BATCH_SIZE, common_utils.DEPTH],
                                tf.shape(states_h))


if __name__ == "__main__":
    tf.test.main()
