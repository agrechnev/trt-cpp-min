TensorRT C++ (almost) minimal examples
====

By Oleksiy Grechnyev, IT-JIM, Mar-Apr 2020.

This is a minimal C++ TensorRT example, much simpler than the tutorial from Nvidia. I create a trivial neural network 
of a single Linear layer (3D -> 2D output) in PyTorch, convert in to ONNX, and run in C++ TensorRT. Requires CUDA and
TensorRT (`libnvinfer`, `libnvonnxparser`) installed in your system.

`model1` weights and biases:  
w=[[1., 2., 3.], [4., 5., 6.]]  
b=[-1., -2.]  

For example, inferring for x=[0.5, -0.5, 1.0] should give y=[1.5, 3.5]. 

* `gen_model1.py` A python 3 code to create and test model1.onnx. Requires `torch`, `onnx`, `onnxruntime`  
* `example.1` A minimal C++ example 