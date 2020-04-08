# By Olekisy Grechnyev, IT-JIM on 3/30/20.
# Load ONNX model and check
# Infer it with onxruntime

import numpy as np
import onnx
import onnxruntime

#import caffe2.python.onnx.backend as backend

def check_onnx(file_name):
    """Check the ONNX model"""
    print('\nChecking ONNX ' + file_name + ' ...')
    onnx_model = onnx.load(file_name)
    # print('onnx_model =', onnx_model)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))


def infer_onnx(file_name):
    """Infer with onnxruntime"""
    print('\nInferring ONNX with onnxruntime : ' + file_name + ' ...')
    sess = onnxruntime.InferenceSession(file_name)
    print('sess =', sess)
    input, output = sess.get_inputs()[0], sess.get_outputs()[0]
    print('input =', input)
    print('output =', output)
    x = np.array([[0.5, -0.5, 1.0]], dtype='float32')
    y = sess.run([output.name], {input.name: x})
    print('y =', y)  # [[1.5, 3.5]]


def infer_caffe2():
    print('\nInferring ONNX with caffe2 ...')
    onnx_model = onnx.load('model1.onnx')
    rep = backend.prepare(onnx_model, device='CUDA:0') # Does not work !
    print('rep =', rep)

if __name__ == '__main__':
    check_onnx('model1.onnx')
    check_onnx('model2.onnx')
    infer_onnx('model1.onnx')
    infer_onnx('model2.onnx')
    # infer_caffe2()
