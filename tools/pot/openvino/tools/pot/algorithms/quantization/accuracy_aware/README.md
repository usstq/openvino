# AccuracyAwareQuantization Algorithm {#accuracy_aware_README}

## Introduction
AccuracyAwareQuantization algorithm is aimed at accurate quantization and allows the model's accuracy to stay within the 
pre-defined range. This may cause a 
degradation in performance in comparison to [DefaultQuantization](../default/README.md) algorithm because some layers can be reverted back to the original precision.

## Parameters
Since the [DefaultQuantization](../default/README.md) algorithm is used as an initialization, all its parameters are also valid and can be specified. Here is an example of the definition of the `AccuracyAwareQuantization` method and its parameters:
```json
{
    "name": "AccuracyAwareQuantization", // the name of optimization algorithm 
    "params": {
        ...
    }
}
```

Below is the description of AccuracyAwareQuantization-specific parameters:
- `"ranking_subset_size"` - size of a subset that is used to rank layers by their contribution to the accuracy drop. 
Default value is `300`. The more samples it has the better ranking you have, potentially.
- `"max_iter_num"` - maximum number of iterations of the algorithm, in other words maximum number of layers that may
 be reverted back to floating-point precision. By default it is limited by the overall number of quantized layers.
- `"maximal_drop"` - maximum accuracy drop which has to be achieved after the quantization. Default value is `0.01` (1%).
- `"drop_type"` - drop type of the accuracy metric: 
    - `"absolute"` (default) - absolute drop with respect to the results of the full-precision model
    - `"relative"` - relative to the results of the full-precision model
- `"use_prev_if_drop_increase"` - whether to use network snapshot from the previous iteration of in case if drop 
increases. Default value is `True`.
- `"base_algorithm"` - name of the algorithm that is used to quantize model at the beginning. Default value is 
    "DefaultQuantization".
- `"convert_to_mixed_preset"` - whether to convert the model to "mixed" mode if the accuracy criteria for the model
 quantized with "performance" preset are not satisfied. This option can help to reduce number of layers that are reverted
 to floating-point precision. Note: this is an experimental feature.
- `"metrics"` - optional list of metrics that are taken into account during optimization. It consists of tuples with the 
following parameters:
    - `"name"` - name of the metric to optimize
    - `"baseline_value"` - baseline metric value of the original model. This is the optional parameter. The validations on
    the whole validation will be initiated in the beginning if nothing specified.
- `"metric_subset_ratio"` -  part of the validation set that is used to compare original full-precision and 
fully quantized models when creating ranking subset in case of predefined metric values of the original model.
Default value is `0.5`.
- `"tune_hyperparams"` - enables quantization parameters tuning as a preliminary step before reverting layers back
to the floating-point precision. It can bring additional performance and accuracy boost but increase overall 
quantization time. Default value is `False`.

## Examples

Example:
 * [Quantization of Object Detection model with control of accuracy](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/object_detection)

 A template and full specification for AccuracyAwareQuantization algorithm for POT command-line interface:
 * [Template](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/accuracy_aware_quantization_template.json)
 * [Full specification](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot/configs/accuracy_aware_quantization_spec.json)

