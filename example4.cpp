// By Oleksiy Grechnyev, IT-JIM
// Example 4 : Here I construct network in-place, no ONNX parsing !
// I use here dynamic batches, a batch of 2

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include <cuda_runtime.h>

//======================================================================================================================

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) override {
        using namespace std;
        string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        cerr << s << ": " << msg << endl;
    }
};
//======================================================================================================================

/// Using unique_ptr with Destroy is optional, but beats calling destroy() for everything
/// Borrowed from the NVidia tutorial, nice C++ skills !
template<typename T>
struct Destroy {
    void operator()(T *t) const {
        t->destroy();
    }
};

//======================================================================================================================

/// Create model and create a TRT engine
nvinfer1::ICudaEngine *createCudaEngine(nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    };

    // Weights + bias
    vector<float> w0 = {1., 2., 3., 4., 5., 6.};
    vector<float> b0 = {-1., -2.};
    Weights w{DataType::kFLOAT, w0.data(), (int64_t) w0.size()};
    Weights b{DataType::kFLOAT, b0.data(), (int64_t) b0.size()};

    // Note, so hard to make a simple linear layer
    // But they have convolutions, FC etc for images only !!! WTF ???
    ITensor *input = network->addInput("goblin_input", DataType::kFLOAT, Dims2(-1, 3));
    IConstantLayer * const1 = network->addConstant(Dims2(2, 3), w);
    IMatrixMultiplyLayer* mm = network->addMatrixMultiply(*input, MatrixOperation::kNONE, *const1->getOutput(0), MatrixOperation::kTRANSPOSE);
    IConstantLayer * const2 = network->addConstant(Dims2(1, 2), b);
    IElementWiseLayer* ew = network->addElementWise(*mm->getOutput(0), *const2->getOutput(0), ElementWiseOperation::kSUM);
    ITensor *output = ew->getOutput(0);
    output->setName("goblin_output");
    network->markOutput(*output);

    // Create Optimization profile and set the batch size
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("goblin_input", OptProfileSelector::kMIN, Dims2{batchSize, 3});
    profile->setDimensions("goblin_input", OptProfileSelector::kMAX, Dims2{batchSize, 3});
    profile->setDimensions("goblin_input", OptProfileSelector::kOPT, Dims2{batchSize, 3});

    // Build engine
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    return builder->buildEngineWithConfig(*network, *config);
}

//======================================================================================================================
/// Run a single inference
void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, std::vector<float> const &inputTensor,
                     std::vector<float> &outputTensor, void **bindings, int batchSize) {

    int inputId = 0, outputId = 1; // Here I assume input=0, output=1 for the current network

    // Infer asynchronously, in a proper cuda way !
    using namespace std;
    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[outputId], outputTensor.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

//======================================================================================================================
int main() {
    using namespace std;
    using namespace nvinfer1;

    // Parse model, create engine
    Logger logger;
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT example4 !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    int batchSize = 2;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine(logger, batchSize));

    if (!engine)
        throw runtime_error("Engine creation failed !");

    // Optional : Print all bindings : name + dims + dtype
    cout << "=============\nBindings :\n";
    int n = engine->getNbBindings();
    for (int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << " : dims=";
        for (int j = 0; j < d.nbDims; ++j) {
            cout << d.d[j];
            if (j < d.nbDims - 1)
                cout << "x";
        }
        cout << " , dtype=" << (int) engine->getBindingDataType(i) << " ";
        cout << (engine->bindingIsInput(i) ? "IN" : "OUT") << endl;
    }
    cout << "=============\n\n";

    // Create context
    logger.log(ILogger::Severity::kINFO, "Creating context ...");
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context(engine->createExecutionContext());
    // Very important, you must set batch size here, otherwise you get zero output !
    context->setBindingDimensions(0, Dims2(batchSize, 3));

    // Create data structures for the inference
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    vector<float> inputTensor{0.5, -0.5, 1.0, 0.0, 0.0, 0.0};
    vector<float> outputTensor(2 * batchSize, -4.9);
    void *bindings[2]{0};
    // Alloc cuda memory for IO tensors
    size_t sizes[] = {inputTensor.size(), outputTensor.size()};
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], sizes[i] * sizeof(float));
    }

    // Run the inference !
    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
    cudaStreamSynchronize(stream);
    // Must be [ [1.5, 3.5], [-1,-2] ]
    cout << "y = [";
    for (int i = 0; i < batchSize; ++i) {
        cout << " [" << outputTensor.at(2 * i) << ", " << outputTensor.at(2 * i + 1) << "]";
        if (i < batchSize - 1)
            cout << ", ";
    }
    cout << " ]" << endl;

    cudaStreamDestroy(stream);
    return 0;
}
