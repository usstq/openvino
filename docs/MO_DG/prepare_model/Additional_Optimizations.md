# Embedding Preprocessing Computation {#openvino_docs_MO_DG_Additional_Optimization_Use_Cases}

Input data for inference can be different from the training dataset and requires additional preprocessing before inference.
To accelerate the whole pipeline including preprocessing and inference, Model Optimizer provides special parameters such as `--mean_values`,
`--scale_values`, `--reverse_input_channels`, and `--layout`. Based on these parameters, Model Optimizer generates IR with additionally
inserted sub-graph that performs the defined preprocessing. This preprocessing block can perform mean-scale normalization of input data,
reverting data along channel dimension, and changing the data layout. For more details about these parameters, refer to the paragraphs below.
The same functionality is also available in runtime, please refer to [Overview of Preprocessing API](../../OV_Runtime_UG/preprocessing_overview.md)
for more information.

## When to Specify Layout

You may need to set input layouts, as it is required by some preprocessing, for example, setting a batch,
applying mean or scales, and reversing input channels (BGR<->RGB).

Layout defines the meaning of dimensions in shape and can be specified for both inputs and outputs.
For the layout syntax, check the [Layout API overview](../../OV_Runtime_UG/layout_overview.md). 
To specify the layout, you can use `--layout` option followed by the layout value. 

For example, for Tensorflow\* `nasnet_large` model that was exported to ONNX format and thus has input with `NHWC` layout:

```
mo --input_model tf_nasnet_large.onnx --layout nhwc
```

Additionally, if a model has more than one input or needs both input and output layouts specified,
you need to provide the name of each input or output to which you apply the layout.

For example, for ONNX\* `Yolo v3 Tiny` model that has first input `input_1` in `NCHW` layout and second input `image_shape`
with 2 dimensions: batch and size of the image which can be expressed as `N?` layout:

```
mo --input_model yolov3-tiny.onnx --layout input_1(nchw),image_shape(n?)
```

## How to Change Layout of a Model Inputs and Outputs

Changing the model layout may be necessary if it differs from the one presented by input data. 
To change the layout, you can use either `--layout` or `--source_layout` with `--target_layout`.

For example, for the same `nasnet_large` that were mentioned previously we may want to provide data in `NCHW` layout:

```
mo --input_model tf_nasnet_large.onnx --source_layout nhwc --target_layout nchw
mo --input_model tf_nasnet_large.onnx --layout "nhwc->nchw"
```

Again, if a model has more than one input or needs both input and output layouts specified, you need to provide the name of each input or output to which you apply the layout.

For example, to provide data in the `NHWC` layout for the `Yolo v3 Tiny` model mentioned earlier:

```
mo --input_model yolov3-tiny.onnx --source_layout "input_1(nchw),image_shape(n?)" --target_layout "input_1(nhwc)"
mo --input_model yolov3-tiny.onnx --layout "input_1(nchw->nhwc),image_shape(n?)"
```

## When to Specify Mean and Scale Values
Usually neural network models are trained with the normalized input data. This means that the input data values are converted to be in a specific range,
for example, `[0, 1]` or `[-1, 1]`. Sometimes the mean values (mean images) are subtracted from the input data values as part of the pre-processing.
There are two cases of how the input data pre-processing is implemented.
 * The input pre-processing operations are a part of a model. In this case, the application does not pre-process the input data as a separate step: everything is embedded into the model itself.
 * The input pre-processing operations are not a part of a model and the pre-processing is performed within the application which feeds the model with input data.

In the first case, the Model Optimizer generates the IR with required pre-processing operations and no `mean` and `scale` parameters are required.

In the second case, information about mean/scale values should be provided to the Model Optimizer to embed it to the generated IR.
Model Optimizer provides command-line parameters to specify the values: `--mean_values`, `--scale_values`, `--scale`.
Using these parameters, Model Optimizer embeds the corresponding preprocessing block for mean-value normalization of the input data
and optimizes this block so that the preprocessing takes negligible time for inference.

For example, run the Model Optimizer for the PaddlePaddle* UNet model and apply mean-scale normalization to the input data.

```sh
mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255
```

## When to Reverse Input Channels <a name="when_to_reverse_input_channels"></a>
Sometimes input images for your application can be of the RGB (BGR) format and the model is trained on images of the BGR (RGB) format,
the opposite color channel order. In this case, it is important to preprocess the input images by reverting the color channels before inference.
To embed this preprocessing step into IR, Model Optimizer provides the `--reverse_input_channels` command-line parameter to shuffle the color channels.

The `--reverse_input_channels` parameter applies to an input of the model in two cases.
 * Only one dimension in the input shape has a size equal to 3.
 * One dimension has an undefined size and is marked as `C` channel using `layout` parameters.

Using the `--reverse_input_channels` parameter, Model Optimizer embeds the corresponding preprocessing block for reverting
the input data along channel dimension and optimizes this block so that the preprocessing takes negligible time for inference.

For example, launch the Model Optimizer for the TensorFlow* AlexNet model and embed `reverse_input_channel` preprocessing block into IR.

```sh
mo --input_model alexnet.pb --reverse_input_channels
```

> **NOTE**: If both mean and scale values are specified, the mean is subtracted first and then the scale is applied regardless of the order of options
in the command line. Input values are *divided* by the scale value(s). If also `--reverse_input_channels` option is used, the `reverse_input_channels`
will be applied first, then `mean` and after that `scale`. The data flow in the model looks as follows:
`Parameter -> ReverseInputChannels -> Mean apply-> Scale apply -> the original body of the model`.

## See Also
* [Overview of Preprocessing API](../../OV_Runtime_UG/preprocessing_overview.md)
