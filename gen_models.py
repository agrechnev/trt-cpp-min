# By Olekisy Grechnyev, IT-JIM on 3/30/20.
# Generate a trivial model in pytorch
# Convert to ONNX

import torch
import numpy as np

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
    model.eval()
    x = torch.randn(1, 3, requires_grad=True, device='cuda')
    
    #Export model1.onnx with batch_size=1 
    print('\nExporting model1.onnx ...')
    torch.onnx.export(model,
                      x,
                      'model1.onnx',
                      opset_version=9,
                      verbose=True,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      #dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    #'output': {0: 'batch_size'}}
                      )
    
    #Export model2.onnx with dynamic batch_size
    print('\nExporting model2.onnx ...')
    torch.onnx.export(model,
                      x,
                      'model2.onnx',
                      #opset_version=11,
                      verbose=True,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}}
                      )


if __name__ == '__main__':
    main()
