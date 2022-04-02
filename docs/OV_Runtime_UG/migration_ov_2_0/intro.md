# Transition to OpenVINO™ 2.0 {#openvino_2_0_transition_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_2_0_deployment
   openvino_2_0_inference_pipeline
   openvino_2_0_configure_devices
   openvino_2_0_preprocessing
   openvino_2_0_model_creation

@endsphinxdirective

### Introduction

Older versions of OpenVINO™ (prior to 2022.1) required changes in the application logic when migrating an app from other frameworks, like TensorFlow, ONNX Runtime, PyTorch, PaddlePaddle, etc. The changes were required because:

- Model Optimizer changed input precisions for some inputs. For example, neural language processing models with `I64` inputs were changed to include `I32` ones.
- Model Optimizer changed layouts for TensorFlow models (see [Layouts in OpenVINO](../layout_overview.md)). It lead to unusual requirement of using the input data with a different layout than that of the framework:
![tf_openvino]
- Inference Engine API (`InferenceEngine::CNNNetwork`) applied some conversion rules for input and output precisions due to limitations in device plugins.
- Users needed to specify input shapes during model conversions in Model Optimizer and work with static shapes in the application.

The new OpenVINO™ introduces API 2.0 (also called OpenVINO API v2) to align the logic of working with models as it is done in their origin frameworks - no layout and precision changes, operating with tensor names and indices to address inputs and outputs. OpenVINO Runtime has combined the Inference Engine API used for inference and the nGraph API targeted to work with models and operations. API 2.0 has a common structure, naming convention styles, namespaces, and removes duplicated structures. See [Changes to Inference Pipeline in OpenVINO API v2](common_inference_pipeline.md) for details.

> **NOTE**: Your existing applications will continue to work with OpenVINO Runtime 2022.1, as normal. Although, we strongly recommend migrating to API 2.0 to unlock additional features, like [Preprocessing](../preprocessing_overview.md) and [Dynamic shapes support](../ov_dynamic_shapes.md).

### Introducing IR v11

To support these features, OpenVINO has introduced IR v11, which is now the default version for Model Optimizer. The model represented in IR v11 fully matches the original model in the original framework format in terms of inputs and outputs. Also, you do not have to specify input shapes during conversion, so the resulting IR v11 contains `-1` to denote undefined dimensions. See [Working with dynamic shapes](../ov_dynamic_shapes.md) to fully utilize this feature, or [Changing input shapes](../ShapeInference.md) to reshape to static shapes in the application.

Note that IR v11 is fully compatible with old applications written with the Inference Engine API used by older versions of OpenVINO. It is due to additional runtime information included in IR v11, allowing backward compatibility. This means that if the IR v11 is read by an application based on Inference Engine, it is internally converted to IR v10.

IR v11 is supported by all OpenVINO Development tools including Post-Training Optimization tool, Benchmark app, etc.

### IR v10 Compatibility

API 2.0 also supports models in IR v10 for backward compatibility. If you have IR v10 files, they can be fed to OpenVINO Runtime as well (see [migration steps](common_inference_pipeline.md)).

Some OpenVINO Development Tools also support both IR v10 and IR v11 as an input:
- Accuracy checker uses API 2.0 for model accuracy measurement by default, but also supports switching to the old API using the `--use_new_api False` command line parameter. Both launchers accept IR v10 and v11, but in some cases configuration files should be updated. More details can be found in [Accuracy Checker documentation](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/launcher/openvino_launcher_readme.md).
- [Compile tool](../../../tools/compile_tool/README.md) compiles the model to be used in API 2.0 by default. If you want to use the resulting compiled blob under the Inference Engine API, the additional `ov_api_1_0` option should be passed.

The following OpenVINO tools don't support IR v10 as an input and require the latest version of Model Optimizer to generate IR v11 files:
- Post-Training Optimization tool
- Deep Learning Workbench

> **NOTE**: If you need to quantize your IR v10 models to run with OpenVINO 2022.1, it is recommended to download and use Post-Training Optimization tool from the OpenVINO 2021.4 release.

### Differences between Inference Engine and OpenVINO Runtime 2022.1

Inference Engine and nGraph APIs are not deprecated, they are fully functional and can be used in applications. However, it is highly recommended to migrate to API 2.0, because it offers additional features, which will be further extended in future releases. The following list of additional features is supported by API 2.0:
- [Working with dynamic shapes](../ov_dynamic_shapes.md). The feature increases performance when working with compatible models, such as NLP (Neural Language Processing) and super-resolution models.
- [Preprocessing of the model](../preprocessing_overview.md). The function adds preprocessing operations to inference models and fully occupies the accelerator, freeing CPU resources.

To define the API differences between Inference Engine and API 2.0, let's define two types of behaviors:
- **Old behavior** of OpenVINO supposes:
  - Model Optimizer can change input element types, order of dimensions (layouts), for the model from the original framework.
  - Inference Engine can override input and output element types.
  - Inference Engine API uses operation names to address inputs and outputs (e.g. InferenceEngine::InferRequest::GetBlob).
  - Inference Engine API does not support compiling of models with dynamic input shapes.
- **New behavior** assumes full model alignment with the framework and is implemented in OpenVINO 2022.1:
  - Model Optimizer preserves input element types, order of dimensions (layouts), and stores tensor names from the original models.
  - OpenVINO Runtime 2022.1 reads models in any format (IR v10, IR v11, ONNX, PaddlePaddle, etc.).
  - API 2.0 uses tensor names. Note, the difference between tensor names and operation names is that if a single operation has several output tensors, such tensors cannot be identified in a unique manner, so tensor names are used for addressing as it's usually done in the frameworks.
  - API 2.0 can address input and output tensors also by the index. Some model formats like ONNX are sensitive to the input and output order, which is preserved by OpenVINO 2022.1.

The table below demonstrates which behavior, **old** or **new**, is used for models based on the two APIs.

|               API             | IR v10  | IR v11  | ONNX file | Model created in code |
|-------------------------------|---------|---------|-----------|-----------------------|
|Inference Engine / nGraph APIs |     Old |     Old |       Old |                   Old |
|API 2.0                        |     Old |     New |       New |                   New |

Check these transition guides to understand how to migrate Inference Engine-based applications to API 2.0:
 - [Installation & Deployment](deployment_migration.md)
 - [OpenVINO™ Common Inference pipeline](common_inference_pipeline.md)
 - [Preprocess your model](./preprocessing.md)
 - [Configure device](./configure_devices.md)
 - [OpenVINO™ Model Creation](graph_construction.md)

[tf_openvino]: ../../img/tf_openvino.png
