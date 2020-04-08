// By Oleksiy Grechnyev, IT-JIM
// Example 3-save : Like example 2, divided into 'save' and load 'parts'
// I use here batch of 2

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

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
nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    };
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
            nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    // Create Optimization profile and set the batch size
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims2{batchSize, 3});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims2{batchSize, 3});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims2{batchSize, 3});

    // Build engine
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    // This is needed for TensorRT 6, not needed by 7 !
    config->setMaxWorkspaceSize(64*1024*1024);
    config->addOptimizationProfile(profile);
    return builder->buildEngineWithConfig(*network, *config);
}

//======================================================================================================================
int main() {
    using namespace std;
    using namespace nvinfer1;

    // Parse model, create engine
    Logger logger;
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT example3-save !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    int batchSize = 2;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("model2.onnx", logger, batchSize));

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

    // Write engine to disk
    unique_ptr<IHostMemory, Destroy<IHostMemory>> serializedEngine(engine->serialize());
    cout << "\nSerialized engine : size = " << serializedEngine->size() << ", dtype = " << (int) serializedEngine->type()
         << endl;

    ofstream out("example3.engine", ios::binary);
    out.write((char *)serializedEngine->data(), serializedEngine->size());

    return 0;
}
