// By Oleksiy Grechnyev, IT-JIM
// Example trt6_1: Here I try to load a 'dynamic batch' model in TensorRT 6

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include <NvInfer.h>
#include <NvOnnxParser.h>

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

/// Parse onnx file and create a TRT engine
nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};

    builder->setMaxBatchSize(2);

    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

//    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{builder->createNetwork()};

    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
            nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    // Modern version with config
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    // This is needed for TensorRT 6, not needed by 7 !
    config->setMaxWorkspaceSize(64*1024*1024);
    return builder->buildEngineWithConfig(*network, *config);
}

//======================================================================================================================
/// Run a single inference
void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, std::vector<float> const &inputTensor,
                     std::vector<float> &outputTensor, void **bindings, int batchSize) {

    int inputId = 0, outputId = 1; // Here I assume input=0, output=1 for the current network

    // Infer synchronously as an alternative, no stream needed
//    cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
//    bool res = context->execute(batchSize, bindings);
//    cudaMemcpy(outputTensor.data(), bindings[outputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Infer asynchronously, in a proper cuda way !
    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    context->enqueue(batchSize, bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[outputId], outputTensor.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

//======================================================================================================================
int main() {
    using namespace std;
    using namespace nvinfer1;

    // Parse model, create engine
    Logger logger;
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT 6 example_trt6_1 !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("model1.onnx", logger));

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

    // Create data structures for the inference
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int batchSize = 1;
    vector<float> inputTensor{0.5, -0.5, 1.0};
//    vector<float> inputTensor{0.5, -0.5, 1.0, 0.0, 0.0, 0.0};
    vector<float> outputTensor(2*batchSize, -4.9);
    void *bindings[2]{0};
    // Alloc cuda memory for IO tensors
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        Dims dims{engine->getBindingDimensions(i)};
        size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], size * sizeof(float));
    }

    // Run the inference !
    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
    cudaStreamSynchronize(stream);
    // Must be [1.5, 3.5]
//    cout << "y = [" << outputTensor[0] << ", " << outputTensor[1] << "]" << endl;

    cout << "y = [";
    for (int i = 0; i < batchSize; ++i) {
        cout << " [" << outputTensor.at(2 * i) << ", " << outputTensor.at(2 * i + 1) << "]";
        if (i < batchSize - 1)
            cout << ", ";
    }
    cout << " ]" << endl;

    cudaStreamDestroy(stream);
    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
    return 0;
}
