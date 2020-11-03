//
// Created by liushuai on 2020/11/3.
//

#ifndef INFERENCE_UFFFASTERRCNN_H
#define INFERENCE_UFFFASTERRCNN_H
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "frcnnUtils.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

//! \class
//!
//! \brief The class that defines the overall workflow of this sample.
//!
class UffFasterRcnn
{
    template <typename T>
    using UniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    UffFasterRcnn(const UffFasterRcnnParams& params)
            : mParams(params)
            , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    std::vector<vPPM> ppms;

    UffFasterRcnnParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an UFF model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
                          UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvuffparser::IUffParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Helper function to do post-processing(apply delta to ROIs).
    //!
    void batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
                                            const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
                                            std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img, int N);

    //!
    //! \brief NMS helper function in post-processing.
    //!
    std::vector<int> nms_classifier(std::vector<float>& boxes_per_cls, std::vector<float>& probs_per_cls,
                                    float NMS_OVERLAP_THRESHOLD, int NMS_MAX_BOXES);

    //!
    //! \brief Helper function to dump bbox-overlayed images as PPM files.
    //!
    void visualize_boxes(int img_num, int class_num, std::vector<float>& pred_boxes, std::vector<float>& pred_probs,
                         std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img, std::vector<vPPM>& ppms);
};


#endif //INFERENCE_UFFFASTERRCNN_H
