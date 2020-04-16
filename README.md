TensorRT 7 C++ (almost) minimal examples
====

By Oleksiy Grechnyev, IT-JIM, Mar-Apr 2020.

### Introduction

`example1` is a minimal C++ TensorRT 7 example, much simpler than Nvidia examples. I create a trivial neural network 
of a single Linear layer (3D -> 2D output) in PyTorch, convert in to ONNX, and run in C++ TensorRT 7. Requires CUDA and
TensorRT 7 (`libnvinfer`, `libnvonnxparser`) installed in your system. Other examples are not much harder.

Note : These examples are for TensorRT 7+ only (see discussion below on TensorRT 6). A lot has changed in this version, especially compared to TensorRT 5 !
ONNX with dynamic batch size is now difficult.
You must set the optimization profile, min/max/opt input size, and finally actual input size (in the context).
Here I use `model1.onnx` with fixed batch size in `example1`, and `model2.onnx` with dynamic batch size in `example2`.  

`model1`, `model2` weights and biases:  
w=[[1., 2., 3.], [4., 5., 6.]]  
b=[-1., -2.]  

For example, inferring for x=[0.5, -0.5, 1.0] should give y=[1.5, 3.5].

### Experiments with TensorRT 6:

I tried to run this with TensorRT 6 in docker and discovered the following issues:  
1. Parser does not like ONNX generated with PyTorch > 1.2, re-generated models on PyTorch 1.2  
2. The code does not run without an extra line `config->setMaxWorkspaceSize(...);`  
3. At this point, examples 1, 4, 5 work fine, but not 2, 3 (Parse ONNX with dynamic batch size)
4. However, now `example1` can infer `model2.onnx` (only with batch_size = 1), which did not work on TensorRT 7

My investigation showed that TensorRT 6 internally has all the dynamic dimension infrastructure
(dim=-1, optimization profiles), but the ONNX parser cannot parse the ONNX network with the dynamic dimension!
It just throws away the batch dimension (it is removed, not set to 1). As the result, you can infer such network
as in `example1`, and only with batch_size = 1.

Update: This was with the "explicit batch" (`kEXPLICIT_BATCH`) option in the model definition. What does this mean?  
Apparently, this option means that network has an explicit batch dimension (which can be 1 or -1 or something else).  

* TensorRT 7 : Without `kEXPLICIT_BATCH`, ONNX cannot be parsed  
* TensorRT 6 : With `kEXPLICIT_BATCH`, ONNX parser does not support dynamic dimensions, and even without them it tends to misbehave
for many networks. However, with TensorRT 6 you can parse ONNX without `kEXPLICIT_BATCH`. This works fine in TensorRT 6, but not 7!  

### Examples

* `gen_models.py` A python 3 code to create `model1.onnx` and `model2.onnx`. Requires `torch`  
* `check_models.py` A python 3 code to check and test `model1.onnx` and `model2.onnx`. Requires `numpy`, `onnx`, `onnxruntime`  
* `example1` A minimal C++ example, runs `model1.onnx` (with fixed batch size of 1)  
* `example2` Runs `model2.onnx` (with dynamic batch size)   
* `example3` Serialization: like `example2`, but split into save and load parts  
* `example4` Create simple network in-place (no ONNX parsing)  
* `example5` Another in-place network with FullyConnected layer, and tried INT8 quantization (but it fails for this layer, it seems). FP16 works fine though.
* `example6` Convolution layer example
* `example7` Finally succeeded with INT8 using a conv->relu->conv->relu network  
