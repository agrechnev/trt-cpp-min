// By Oleksiy Grechnyev, IT-JIM
// Example 7 : Finally I succeeded with int8 (if your GPU supports it)
// It seems only combinations conv+relu are available
// Also, for some reason, one layer is not enough, so I used conv->relu->conv->relu network
// Update : added inference

#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <NvInfer.h>

#include <cuda_runtime.h>

constexpr bool USE_INT8 = true;

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
/// Optional : Print dimensions as string
std::string printDim(const nvinfer1::Dims &d) {
    using namespace std;
    ostringstream oss;
    for (int j = 0; j < d.nbDims; ++j) {
        oss << d.d[j];
        if (j < d.nbDims - 1)
            oss << "x";
    }
    return oss.str();
}

//======================================================================================================================
/// Optional : Print layers of the network
void printNetwork(const nvinfer1::INetworkDefinition &net) {
    using namespace std;
    using namespace nvinfer1;
    cout << "\n=============\nNetwork info :" << endl;

    cout << "\nInputs : " << endl;
    for (int i = 0; i < net.getNbInputs(); ++i) {
        ITensor *inp = net.getInput(i);
        cout << "input" << i << " , dtype=" << (int) inp->getType() << " , dims=" << printDim(inp->getDimensions())
             << endl;
    }

    cout << "\nLayers : " << endl;
    cout << "Number of layers : " << net.getNbLayers() << endl;
    for (int i = 0; i < net.getNbLayers(); ++i) {
        ILayer *l = net.getLayer(i);
        cout << "layer" << i << " , name=" << l->getName() << " , type=" << (int) l->getType() << " , IN ";
        for (int j = 0; j < l->getNbInputs(); ++j)
            cout << "(" << int(l->getInput(j)->getType()) << ") " << printDim(l->getInput(j)->getDimensions()) << " ";
        cout << ", OUT ";
        for (int j = 0; j < l->getNbOutputs(); ++j)
            cout << "(" << int(l->getOutput(j)->getType()) << ") " << printDim(l->getOutput(j)->getDimensions()) << " ";
        cout << endl;
        switch (l->getType()) {
            case (LayerType::kCONVOLUTION): {
                IConvolutionLayer *lc = static_cast<IConvolutionLayer *>(l);
                cout << "CONVOLUTION : ";
                cout << "ker=" << printDim(lc->getKernelSizeNd());
                cout << ", stride=" << printDim(lc->getStrideNd());
                cout << ", padding=" << printDim(lc->getPaddingNd());
                cout << ", groups=" << lc->getNbGroups();
                Weights w = lc->getKernelWeights();
                cout << ", weights=" << w.count << ":" << int(w.type);
                cout << endl;
            }
                break;
            case (LayerType::kSCALE): {
                IScaleLayer *ls = static_cast<IScaleLayer *>(l);
                cout << "SCALE: ";
                cout << "mode=" << int(ls->getMode());
                Weights ws = ls->getScale();
                cout << ", scale=" << ws.count << ":" << int(ws.type);
                Weights wp = ls->getPower();
                cout << ", power=" << wp.count << ":" << int(wp.type);
                Weights wf = ls->getShift();
                cout << ", shift=" << wf.count << ":" << int(wf.type);
                cout << endl;
            }
                break;
        }
    }

    cout << "\nOutputs : " << endl;
    for (int i = 0; i < net.getNbOutputs(); ++i) {
        ITensor *outp = net.getOutput(i);
        cout << "input" << i << " , dtype=" << (int) outp->getType() << " , dims=" << printDim(outp->getDimensions())
             << endl;
    }

    cout << "=============\n" << endl;
}
//======================================================================================================================

/// Create model and create a TRT engine
nvinfer1::ICudaEngine *createCudaEngine(nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    };

    // Network
    ITensor *input = network->addInput("goblin_input", DataType::kFLOAT, Dims4(1, 3, 224, 224));

    // conv1
    vector<float> wwC1(7 * 7 * 3 * 64, 0.0123);  // 3x3 box filter
    vector<float> bbC1(64, 0.5);
    Weights wC1{DataType::kFLOAT, wwC1.data(), (int64_t) wwC1.size()};
    Weights bC1{DataType::kFLOAT, bbC1.data(), (int64_t) bbC1.size()};
    IConvolutionLayer *conv1 = network->addConvolutionNd(*input, 64, Dims2(7, 7), wC1, bC1);
    conv1->setStrideNd(Dims2(2, 2));
    conv1->setPaddingNd(Dims2(3, 3));
    // relu1
    IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

    // conv2
    vector<float> wwC2(3 * 3 * 64 * 128, 0.01231);  // 3x3 box filter
    vector<float> bbC2(128, 0.4);
    Weights wC2{DataType::kFLOAT, wwC2.data(), (int64_t) wwC2.size()};
    Weights bC2{DataType::kFLOAT, bbC2.data(), (int64_t) bbC2.size()};
    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), 128, Dims2(3, 3), wC2, bC2);
    conv1->setStrideNd(Dims2(2, 2));
    conv1->setPaddingNd(Dims2(1, 1));
    // relu2
    IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

    ITensor *output = relu2->getOutput(0);
    output->setName("goblin_output");
    network->markOutput(*output);

    printNetwork(*network);

    // Are fancy types available ?
    cout << "platformHasFastFp16 = " << builder->platformHasFastFp16() << endl;
    cout << "platformHasFastInt8 = " << builder->platformHasFastInt8() << endl;

    // Set up the config
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    // This is needed for TensorRT 6, not needed by 7 !
    config->setMaxWorkspaceSize(1024 * 1024 * 1024);

    if (USE_INT8) {
        // Int8 quantization with the explicit range
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);

        // Set the dynamic range for all layers and input
        float minRange = -17., maxRange = 17.;
        cout << "layers = " << network->getNbLayers() << endl;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            ILayer *layer = network->getLayer(i);
            ITensor *tensor = layer->getOutput(0);
            tensor->setDynamicRange(minRange, maxRange);
            layer->setPrecision(DataType::kINT8);
            layer->setOutputType(0, DataType::kINT8);
        }
        network->getInput(0)->setDynamicRange(minRange, maxRange);
    }

    // Build engine
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
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT example7 !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    int batchSize = 1;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine(logger, batchSize));

    if (!engine)
        throw runtime_error("Engine creation failed !");

    // Optional : Print all bindings : name + dims + dtype
    cout << "=============\nBindings :\n";
    int n = engine->getNbBindings();
    for (int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << " : dims=" << printDim(d);
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
    vector<float> inputTensor(3*224*224*batchSize, 3.1);
    vector<float> outputTensor(128*108*108 * batchSize, -4.9);
    for (int iy = 0; iy < 224; ++iy) {
        for (int ix = 0; ix < 224; ++ix) {
            for (int j = 0; j < 3; ++j) {
                inputTensor[iy*224*3 + ix*3 + j] = (ix + iy) % 2;
            }
        }
    }
    cout << "input = " << endl;
    for (int iy = 0; iy < 10; ++iy) {
        for (int ix = 0; ix < 10; ++ix) {
            cout << inputTensor[iy*224*3 + ix*3] << " ";
        }
        cout << endl;
    }

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

    cout << "output = " << endl;
    for (int iy = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix) {
            cout << outputTensor[iy*108*128 + ix*128] << " ";
        }
        cout << endl;
    }

    cudaStreamDestroy(stream);
    cudaFree(bindings[0]);
    cudaFree(bindings[1]);

    return 0;
}
