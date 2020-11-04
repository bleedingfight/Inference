//
// Created by liushuai on 2020/11/4.
//

#ifndef INFERENCE_UFFSSDPARAMS_H
#define INFERENCE_UFFSSDPARAMS_H
struct UffSSDParams : public samplesCommon::SampleParams {
    std::string uffFileName;    //!< The file name of the UFF model to use
    std::string labelsFileName; //!< The file namefo the class labels
    int32_t outputClsSize;      //!< The number of output classes
    int32_t calBatchSize;       //!< The size of calibration batch
    int32_t nbCalBatches;       //!< The number of batches for calibration
    int32_t keepTopK;           //!< The maximum number of detection post-NMS
    float
            visualThreshold; //!< The minimum score threshold to consider a detection
};
#endif //INFERENCE_UFFSSDPARAMS_H
