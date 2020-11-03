//
// Created by liushuai on 2020/11/3.
//

#include "UffFasterRcnn.h"
bool UffFasterRcnn::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    if (mParams.loadEngine.size() > 0)
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
        {
            infer->setDLACore(mParams.dlaCore);
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
                infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        infer->destroy();
        sample::gLogInfo << "TRT Engine loaded from: " << mParams.loadEngine << std::endl;
        if (!mEngine)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

    if (mParams.dlaCore >= 0)
    {
        builder->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        builder->setDLACore(mParams.dlaCore);
        builder->allowGPUFallback(true);
    }
    if (!builder)
    {
        return false;
    }

    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());

    if (!network)
    {
        return false;
    }

    auto parser = UniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, parser);

    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);
    assert(network->getNbOutputs() == 3);
    return true;
}

bool UffFasterRcnn::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
                                     UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvuffparser::IUffParser>& parser)
{
    parser->registerInput(mParams.inputNodeName.c_str(),
                          DimsCHW(mParams.inputChannels, mParams.inputHeight, mParams.inputWidth), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputRegName.c_str());
    parser->registerOutput(mParams.outputClsName.c_str());
    parser->registerOutput(mParams.outputProposalName.c_str());
    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);

    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(2_GiB);
    if (mParams.fp16)
    {
        builder->setFp16Mode(true);
    }
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const std::string listFileName = "list.txt";
        const int imageC = 3;
        const int imageH = mParams.inputHeight;
        const int imageW = mParams.inputWidth;
        nvinfer1::DimsNCHW imageDims{mParams.calBatchSize, imageC, imageH, imageW};
        // To prevent compiler initialization warning with some versions of gcc
        for (int i=imageDims.nbDims; i < Dims::MAX_DIMS; ++i){
            imageDims.d[i] = 0;
            imageDims.type[i] = DimensionType::kSPATIAL;
        }
        BatchStream calibrationStream(
                mParams.calBatchSize, mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(
                new Int8EntropyCalibrator2(calibrationStream, 0, "UffFasterRcnn", mParams.inputNodeName.c_str()));
        builder->setInt8Mode(true);
        // Fallback to FP16 if there is no INT8 kernels.
        builder->setFp16Mode(true);
        builder->setInt8Calibrator(calibrator.get());
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }

    if (mParams.saveEngine.size() > 0)
    {
        std::ofstream p(mParams.saveEngine, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = mEngine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.saveEngine << std::endl;
    }

    return true;
}

bool UffFasterRcnn::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    SimpleProfiler profiler("FasterRCNN performance");

    if (mParams.profile)
    {
        context->setProfiler(&profiler);
    }

    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status{true};

    for (int i = 0; i < mParams.repeat; ++i)
    {
        status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }
    }

    if (mParams.profile)
    {
        std::cout << profiler;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return status;
}

bool UffFasterRcnn::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

bool UffFasterRcnn::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mParams.inputChannels;
    const int inputH = mParams.inputHeight;
    const int inputW = mParams.inputWidth;
    const int batchSize = mParams.batchSize;
    std::vector<std::string> imageList = mParams.inputImages;
    ppms.resize(batchSize);
    assert(ppms.size() <= imageList.size());

    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(imageList[i], ppms[i], mParams.dataDirs);
        // resize to input dimensions.
        resizePPM(ppms[i], inputW, inputH);
    }

    // subtract image channel mean
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputNodeName));
    float pixelMean[3]{103.939, 116.779, 123.68};

    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(ppms[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
            }
        }
    }

    return true;
}

bool UffFasterRcnn::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;
    const int outputClassSize = mParams.outputClassSize;
    std::vector<float> classifierRegressorStd;
    std::vector<std::string> classNames;
    const float* out_class = static_cast<const float*>(buffers.getHostBuffer(mParams.outputClsName));
    const float* out_reg = static_cast<const float*>(buffers.getHostBuffer(mParams.outputRegName));
    const float* out_proposal = static_cast<const float*>(buffers.getHostBuffer(mParams.outputProposalName));
    // host memory for outputs
    std::vector<float> pred_boxes;
    std::vector<int> pred_cls_ids;
    std::vector<float> pred_probs;
    std::vector<int> box_num_per_img;

    int post_nms_top_n = mParams.postNmsTopN;
    // post processing for stage 2.
    batch_inverse_transform_classifier(out_proposal, post_nms_top_n, out_class, out_reg, pred_boxes, pred_cls_ids,
                                       pred_probs, box_num_per_img, batchSize);
    visualize_boxes(batchSize, outputClassSize, pred_boxes, pred_probs, pred_cls_ids, box_num_per_img, ppms);
    return true;
}

//! \brief Define the function to apply delta to ROIs
//!
void UffFasterRcnn::batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
                                                       const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
                                                       std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img, int N)
{
    auto max_index = [](const float* start, const float* end) -> int {
        float max_val = start[0];
        int max_pos = 0;

        for (int i = 1; start + i < end; ++i)
        {
            if (start[i] > max_val)
            {
                max_val = start[i];
                max_pos = i;
            }
        }

        return max_pos;
    };
    int box_num;

    for (int n = 0; n < N; ++n)
    {
        box_num = 0;

        for (int i = 0; i < roi_num_per_img; ++i)
        {
            auto max_idx = max_index(
                    classifier_cls + n * roi_num_per_img * mParams.outputClassSize + i * mParams.outputClassSize,
                    classifier_cls + n * roi_num_per_img * mParams.outputClassSize + i * mParams.outputClassSize
                    + mParams.outputClassSize);

            if (max_idx == (mParams.outputClassSize - 1)
                || classifier_cls[n * roi_num_per_img * mParams.outputClassSize + max_idx + i * mParams.outputClassSize]
                   < mParams.visualizeThreshold)
            {
                continue;
            }

            // inverse transform
            float tx, ty, tw, th;
            tx = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize
                                 + max_idx * 4]
                 / mParams.classifierRegressorStd[0];
            ty = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                                 + 1]
                 / mParams.classifierRegressorStd[1];
            tw = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                                 + 2]
                 / mParams.classifierRegressorStd[2];
            th = classifier_regr[n * roi_num_per_img * mParams.outputBboxSize + i * mParams.outputBboxSize + max_idx * 4
                                 + 3]
                 / mParams.classifierRegressorStd[3];
            float y = roi_after_nms[n * roi_num_per_img * 4 + 4 * i] * static_cast<float>(mParams.inputHeight);
            float x = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 1] * static_cast<float>(mParams.inputWidth);
            float ymax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 2] * static_cast<float>(mParams.inputHeight);
            float xmax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 3] * static_cast<float>(mParams.inputWidth);
            float w = xmax - x;
            float h = ymax - y;
            float cx = x + w / 2.0f;
            float cy = y + h / 2.0f;
            float cx1 = tx * w + cx;
            float cy1 = ty * h + cy;
            float w1 = std::round(std::exp(static_cast<double>(tw)) * w * 0.5f) * 2.0f;
            float h1 = std::round(std::exp(static_cast<double>(th)) * h * 0.5f) * 2.0f;
            float x1 = std::round((cx1 - w1 / 2.0f) * 0.5f) * 2.0f;
            float y1 = std::round((cy1 - h1 / 2.0f) * 0.5f) * 2.0f;
            auto clip
                    = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
            float x2 = x1 + w1;
            float y2 = y1 + h1;
            x1 = clip(x1, 0.0f, mParams.inputWidth - 1.0f);
            y1 = clip(y1, 0.0f, mParams.inputHeight - 1.0f);
            x2 = clip(x2, 0.0f, mParams.inputWidth - 1.0f);
            y2 = clip(y2, 0.0f, mParams.inputHeight - 1.0f);

            if (x2 > x1 && y2 > y1)
            {
                pred_boxes.push_back(x1);
                pred_boxes.push_back(y1);
                pred_boxes.push_back(x2);
                pred_boxes.push_back(y2);
                pred_probs.push_back(classifier_cls[n * roi_num_per_img * mParams.outputClassSize + max_idx
                                                    + i * mParams.outputClassSize]);
                pred_cls_ids.push_back(max_idx);
                ++box_num;
            }
        }

        box_num_per_img.push_back(box_num);
    }
}

//! \brief NMS on CPU in post-processing of classifier outputs.
//!
std::vector<int> UffFasterRcnn::nms_classifier(std::vector<float>& boxes_per_cls,
                                               std::vector<float>& probs_per_cls, float NMS_OVERLAP_THRESHOLD, int NMS_MAX_BOXES)
{
    int num_boxes = boxes_per_cls.size() / 4;
    std::vector<std::pair<float, int>> score_index;

    for (int i = 0; i < num_boxes; ++i)
    {
        score_index.push_back(std::make_pair(probs_per_cls[i], i));
    }

    std::stable_sort(score_index.begin(), score_index.end(),
                     [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                         return pair1.first > pair2.first;
                     });
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }

        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };
    std::vector<int> indices;

    for (auto i : score_index)
    {
        const int idx = i.second;
        bool keep = true;

        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(&boxes_per_cls[idx * 4], &boxes_per_cls[kept_idx * 4]);
                keep = overlap <= NMS_OVERLAP_THRESHOLD;
            }
            else
            {
                break;
            }
        }

        if (indices.size() >= static_cast<unsigned>(NMS_MAX_BOXES))
        {
            break;
        }

        if (keep)
        {
            indices.push_back(idx);
        }
    }

    return indices;
}

//! \brief Dump the detection results(bboxes) as PPM images, overlayed on original image.
//!
void UffFasterRcnn::visualize_boxes(int img_num, int class_num, std::vector<float>& pred_boxes,
                                    std::vector<float>& pred_probs, std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
                                    std::vector<vPPM>& ppms)
{
    int box_start_idx = 0;
    std::vector<float> boxes_per_cls;
    std::vector<float> probs_per_cls;
    std::vector<BBox> det_per_img;

    for (int i = 0; i < img_num; ++i)
    {
        det_per_img.clear();

        for (int c = 0; c < (class_num - 1); ++c)
        { // skip the background
            boxes_per_cls.clear();
            probs_per_cls.clear();

            for (int k = box_start_idx; k < box_start_idx + box_num_per_img[i]; ++k)
            {
                if (pred_cls_ids[k] == c)
                {
                    boxes_per_cls.push_back(pred_boxes[4 * k]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 1]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 2]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 3]);
                    probs_per_cls.push_back(pred_probs[k]);
                }
            }

            // apply NMS algorithm per class
            auto indices_after_nms
                    = nms_classifier(boxes_per_cls, probs_per_cls, mParams.nmsIouThresholdClassifier, mParams.postNmsTopN);

            // Show results
            for (unsigned k = 0; k < indices_after_nms.size(); ++k)
            {
                int idx = indices_after_nms[k];
                std::cout << "Detected " << mParams.classNames[c] << " in " << ppms[i].fileName << " with confidence "
                          << probs_per_cls[idx] * 100.0f << "% " << std::endl;
                BBox b{boxes_per_cls[idx * 4], boxes_per_cls[idx * 4 + 1], boxes_per_cls[idx * 4 + 2],
                       boxes_per_cls[idx * 4 + 3]};
                det_per_img.push_back(b);
            }
        }

        box_start_idx += box_num_per_img[i];
        writePPMFileWithBBox(ppms[i].fileName + "_det.ppm", ppms[i], det_per_img);
    }
}