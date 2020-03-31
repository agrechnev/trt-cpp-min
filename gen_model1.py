# By Olekisy Grechnyev, IT-JIM on 3/30/20.
# Generate a trivial model in pytorch
# Convert to ONNX
# Load and check
# Load and infer with onxruntime

import onnx
import onnxruntime

import torch
import numpy as np

#import caffe2.python.onnx.backend as backend


def main():
    """Create a very simple ONNX model"""
    model = torch.nn.Linear(3, 2)

    w, b = model.state_dict()['weight'], model.state_dict()['bias']
    # w, b = model.weight, model.bias
    with torch.no_grad():
        w.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        b.copy_(torch.tensor([-1., -2.]))
    model.cuda()
    print('w = ', w, w.dtype)
    print('b = ', b, b.dtype)

    # Check the standard result, should be [1.5, 3.5]
    x = torch.tensor([0.5, -0.5, 1.0], device='cuda')
    y = model(x)
    print('x =', x)
    print('y =', y)

    # Export to ONNX
    print('Exporting ONNX ...')
    model.eval()
    x = torch.randn(1, 3, requires_grad=True, device='cuda')
    torch.onnx.export(model,
                      x,
                      'model1.onnx',
                      opset_version=11,
                      verbose=True,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      #dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    #'output': {0: 'batch_size'}}
                      )


def check_onnx():
    """Check the ONNX model"""
    print('Checking ONNX ...')
    onnx_model = onnx.load('model1.onnx')
    # print('onnx_model =', onnx_model)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))


def infer_onnx():
    """Infer with onnxruntime"""
    print('Inferring ONNX with onnxruntime')
    sess = onnxruntime.InferenceSession('model1.onnx')
    print('sess =', sess)
    input, output = sess.get_inputs()[0], sess.get_outputs()[0]
    print('input =', input)
    print('output =', output)
    x = np.array([[0.5, -0.5, 1.0]], dtype='float32')
    y = sess.run([output.name], {input.name: x})
    print('y =', y)  # [[1.5, 3.5]]


def infer_caffe2():
    print('Inferring ONNX with caffe2 ...')
    onnx_model = onnx.load('model1.onnx')
    rep = backend.prepare(onnx_model, device='CUDA:0') # Does not work !
    print('rep =', rep)


if __name__ == '__main__':
    main()
    check_onnx()
    infer_onnx()
    # infer_caffe2()
