import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
from keras.models import load_model

with tf.Session() as sess:
    model=load_model('models/Hourglass_modelsV3_epoch7.h5')
    with tf.gfile.FastGFile('models/Hourglass_tf.pb','wb') as f:
        graph_def = sess.graph.as_graph_def()
        output_nodes = []
        input_nodes = []
        for ot in model.outputs:
            output_nodes.append(ot.op.name)
        for it in model.inputs:
            input_nodes.append(it.op.name)
        print('inputs:',input_nodes)
        print('outputs:',output_nodes)
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
        f.write(output_graph_def.SerializeToString())

with tf.gfile.FastGFile('models/Hourglass_tf.pb','rb') as f:
    intput_graph_def=tf.GraphDef()
    intput_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as p_graph:
        tf.import_graph_def(intput_graph_def, name="")

# for i in p_graph.get_operations():
#     if 'input' in i.name:
#         print(i.name)

inputs=p_graph.get_tensor_by_name('input_1:0')
outputs=p_graph.get_tensor_by_name('nstack_2/Tanh:0')

with tf.Session(graph=p_graph) as sess:
    img = (np.random.random((1, 224, 176, 3)) - 0.5) * 2
    y_pred=sess.run(outputs,feed_dict={inputs:img})
    y_pred=np.squeeze(y_pred,0)
    print(y_pred.shape)