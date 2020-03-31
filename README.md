TensorRT 7 C++ (almost) minimal examples
====

By Oleksiy Grechnyev, IT-JIM, Mar-Apr 2020.

This is a minimal C++ TensorRT 7 example, much simpler than the tutorial from Nvidia. I create a trivial neural network 
of a single Linear layer (3D -> 2D output) in PyTorch, convert in to ONNX, and run in C++ TensorRT 7. Requires CUDA and
TensorRT 7 (`libnvinfer`, `libnvonnxparser`) installed in your system.

Note : This is example for TensorRT 7 only. A lot has changed in this version ! ONNX with dynamic batch size is now difficult.
You must set the optimization profile, min/max/opt input size, and finally actual input size (in the context).
Here I use `model1.onnx` with fixed batch size in `example1`, and `model2.onnx` with dynamic batch size in `example2`.  

`model1`, `model2` weights and biases:  
w=[[1., 2., 3.], [4., 5., 6.]]  
b=[-1., -2.]  

For example, inferring for x=[0.5, -0.5, 1.0] should give y=[1.5, 3.5]. 

* `gen_model1.py` A python 3 code to create and test `model1.onnx` and `model2.onnx`. Requires `torch`, `onnx`, `onnxruntime`  
* `example1` A minimal C++ example, runs `model1.onnx` (with fixed batch size of 1)  
* `example2` Runs `model2.onnx` (with dynamic batch size)   
