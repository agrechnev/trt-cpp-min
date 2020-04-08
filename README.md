TensorRT 7 C++ (almost) minimal examples
====

By Oleksiy Grechnyev, IT-JIM, Mar-Apr 2020.

This is a minimal C++ TensorRT 7 example, much simpler than Nvidia examples. I create a trivial neural network 
of a single Linear layer (3D -> 2D output) in PyTorch, convert in to ONNX, and run in C++ TensorRT 7. Requires CUDA and
TensorRT 7 (`libnvinfer`, `libnvonnxparser`) installed in your system.

Note : This is example for TensorRT 7 only (maybe 6, but not 5 !). A lot has changed in this version ! ONNX with dynamic batch size is now difficult.
You must set the optimization profile, min/max/opt input size, and finally actual input size (in the context).
Here I use `model1.onnx` with fixed batch size in `example1`, and `model2.onnx` with dynamic batch size in `example2`.  

Update: It seems this stuff appeared already in TensorRT 6. I didn't try my code though. But in TensorRT 7 old ways to
do things were hard-deprecated (removed).  


`model1`, `model2` weights and biases:  
w=[[1., 2., 3.], [4., 5., 6.]]  
b=[-1., -2.]  

For example, inferring for x=[0.5, -0.5, 1.0] should give y=[1.5, 3.5]. 

* `gen_models.py` A python 3 code to create `model1.onnx` and `model2.onnx`. Requires `torch`  
* `check_models.py` A python 3 code to check and test `model1.onnx` and `model2.onnx`. Requires `numpy`, `onnx`, `onnxruntime`  
* `example1` A minimal C++ example, runs `model1.onnx` (with fixed batch size of 1)  
* `example2` Runs `model2.onnx` (with dynamic batch size)   
* `example3` Serialization: like `example2`, but split into save and load parts  
* `example4` Create simple network in-place (no ONNX parsing)  
* `example5` Tried quantization (but it fails for these layers, it seems)  
