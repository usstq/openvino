# DefaultQuantization Algorithm {#pot_compression_algorithms_quantization_default_README}

## Introduction
DefaultQuantization algorithm is designed to do a fast and, in many cases, accurate quantization. It does not have any control of accuracy metric but provides a lot of knobs that can be used to improve it.

## Parameters
DefaultQuantization algorithm has mandatory and optional parameters. For more details on how to use these parameters please refer to [Best Practices](@ref pot_docs_BestPractices) document. Here is an example of the definition of DefualtQuantization method and its parameters:
```python
{
    "name": "DefaultQuantization", # the name of optimization algorithm 
    "params": {
        ...
    }
}
```

### Mandatory parameters
- `"preset"` - preset which controls the quantization mode (symmetric and asymmetric). It can take two values:
    - `"performance"` (default) - stands for symmetric quantization of weights and activations. This is the most 
    performant across all the HW.
    - `"mixed"` - symmetric quantization of weights and asymmetric quantization of activations. This mode can be useful
    for quantization of NN which has both negative and positive input values in quantizing operations, e.g. 
    non-ReLU based CNN.  
- `"stat_subset_size"` - size of subset to calculate activations statistics used for quantization. The whole dataset 
is used if no parameter specified. We recommend using not less than 300 samples.


### Optional parameters
All other options can be considered as an advanced mode and require deep knowledge of the quantization process. Below
is an overall description of all possible parameters:
- `"model type"` - An optional parameter, needed for additional patterns in the model, default value is None (supported only "Transformer" now)
- `"inplace_statistic"` - An optional parameter, needed for change method collect statistics, reduces the amount of memory consumed, but increases the calibration time
- `"ignored"` - NN subgraphs which should be excluded from the optimization process 
    - `"scope"` - list of particular nodes to exclude
    - `"operations"` - list of operation types to exclude (expressed in OpenVINO IR notation). This list consists of
    the following tuples:
        - `"type"` - type of ignored operation
        - `"attributes"` - if attributes are defined they will be considered during the ignorance. They are defined by
        a dictionary of  `"<NAME>": "<VALUE>"` pairs.
- `"weights"` - this section manually defines quantization scheme for weights and the way to estimate the 
quantization range for that. It worth noting that changing the quantization scheme may lead to inability to infer such
mode on the existing HW.
    - `"bits"` - bit-width, default is 8
    - `"mode"` - quantization mode (symmetric or asymmetric)
    - `"level_low"` - minimum level in the integer range in which we quantize to, default is 0 for unsigned range, -2^(bit-1) - for signed
    - `"level_high"` - maximum level in the integer range in which we quantize to, default is 2^bits-1 for unsigned range, 2^(bit-1)-1 - for signed
    - `"granularity"` - quantization scale granularity and can take the following two values:
        - `"pertensor"` (default) - per-tensor quantization with one scale factor and zero-point
        - `"perchannel"` - per-channel quantization with per-channel scale factor and zero-point
    - `"range_estimator"` - this section describes parameters of range estimator that is used in MinMaxQuantization 
    method to get the quantization ranges and filter outliers based on the collected statistics. These are the parameters 
    that user can vary to get better accuracy results:
        - `"max"` - parameters to estimate top border of quantizing floating-point range:
            - `"type"` - type of the estimator: 
                - `"max"` (default) - estimates the maximum in the quantizing set of value
                - `"quantile"` - estimates the quantile in the quantizing set of value
            - `"outlier_prob"` - outlier probability used in the "quantile" estimator
        - `"min"` - parameters to estimate bottom border of quantizing floating-point range:
            - `"type"` - type of the estimator: 
                - `"min"` (default) - estimates the minimum in the quantizing set of value
                - `"quantile"` - estimates the quantile in the quantizing set of value
            - `"outlier_prob"` - outlier probability used in the "quantile" estimator
- `"activations"` - this section manually defines quantization scheme for activations and the way to estimate the 
quantization range for that. Again, changing the quantization scheme may lead to inability to infer such
mode on the existing HW.
    - `"bits"` - bit-width, default is 8
    - `"mode"` - quantization mode (symmetric or asymmetric)
    - `"level_low"` - minimum level in the integer range in which we quantize to, default is 0 for unsigned range, -2^(bit-1) - for signed
    - `"level_high"` - maximum level in the integer range in which we quantize to, default is 2^bits-1 for unsigned range, 2^(bit-1)-1 - for signed
    - `"granularity"` - quantization scale granularity and can take the following two values:
        - `"pertensor"` (default) - per-tensor quantization with one scale factor and zero-point
        - `"perchannel"` - per-channel quantization with per-channel scale factor and zero-point
    - `"range_estimator"` - this section describes parameters of range estimator that is used in MinMaxQuantization 
    method to get the quantization ranges and filter outliers based on the collected statistics. These are the parameters 
    that user can vary to get better accuracy results:
        - `"preset"` - preset that defines the same estimator both for top and bottom borders of quantizing 
        floating-point range. Possible value is `"quantile"`.
        - `"max"` - parameters to estimate top border of quantizing floating-point range:
            - `"aggregator"` - type of the function used to aggregate statistics obtained with estimator 
            over the calibration dataset to get a value of the top border:
                - `"mean"` (default) - aggregates mean value
                - `"max"` - aggregates max value
                - `"min"` - aggregates min value
                - `"median"` - aggregates median value
                - `"mean_no_outliers"` - aggregates mean value after removal of extreme quantiles
                - `"median_no_outliers"` - aggregates median value after removal of extreme quantiles
                - `"hl_estimator"` - Hodges-Lehmann filter based aggregator
            - `"type"` - type of the estimator:
                - `"max"` (default) - estimates the maximum in the quantizing set of value
                - `"quantile"` - estimates the quantile in the quantizing set of value
            - `"outlier_prob"` - outlier probability used in the "quantile" estimator
        - `"min"` - parameters to estimate bottom border of quantizing floating-point range:
            - `"type"` - type of the estimator: 
                - `"max"` (default) - estimates the maximum in the quantizing set of value
                - `"quantile"` - estimates the quantile in the quantizing set of value
            - `"outlier_prob"` - outlier probability used in the "quantile" estimator
- `"use_layerwise_tuning"` - enables layer-wise fine-tuning of model parameters (biases, Convolution/MatMul weights and FakeQuantize scales) by minimizing the mean squared error between original and quantized layer outputs.
Enabling this option may increase compressed model accuracy, but will result in increased execution time and memory consumption.

## Examples
Tutorials:
* [Quantization of Image Classification model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)
* [Quantization of Object Detection model from Model Zoo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-detection-quantization)
* [Quantization of Segmentation model for mediacal data](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/110-ct-segmentation-quantize)
* [Quantization of BERT for Text Classification](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)

Examples:
* [Quantization of 3D segmentation model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/3d_segmentation)
* [Quantization of Face Detection model](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/face_detection)
* [Quantizatin of speech model for GNA device](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/speech)

Command-line example:
* [Quantization of Image Classification model](https://docs.openvino.ai/latest/pot_configs_examples_README.html) 

A template and full specification for DefaultQuantization algorithm for POT command-line inferface:
* [Template](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/default_quantization_template.json)
* [Full specification](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/default_quantization_spec.json)


