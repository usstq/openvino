# GNA device {#openvino_docs_OV_UG_supported_plugins_GNA}

The Intel® Gaussian & Neural Accelerator (GNA) is a low-power neural coprocessor for continuous inference at the edge.

Intel® GNA is not intended to replace typical inference devices such as the
CPU, graphics processing unit (GPU), or vision processing unit (VPU). It is designed for offloading
continuous inference workloads including but not limited to noise reduction or speech recognition
to save power and free CPU resources.

The GNA plugin provides a way to run inference on Intel® GNA, as well as in the software execution mode on CPU.

For more details on how to configure a machine to use GNA plugin, see [GNA configuration page](@ref openvino_docs_install_guides_configurations_for_intel_gna).

## Intel® GNA Generational Differences

The first (1.0) and second (2.0) versions of Intel® GNA found in 10th and 11th generation Intel® Core™ Processors may be considered to be functionally equivalent.  Intel® GNA 2.0 provided performance improvement with respect to Intel® GNA 1.0.  Starting with 12th Generation Intel® Core™ Processors (formerly codenamed Alder Lake), support for Intel® GNA 3.0 features is being added.

In the rest of this documentation, "GNA 2.0" refers to Intel® GNA hardware delivered on 10th and 11th generation Intel® Core™ processors, and the term "GNA 3.0" refers to GNA hardware delivered on 12th generation Intel® Core™ processors.

### Intel® GNA Forward and Backward Compatibility

When you run a model using the GNA plugin, it is compiled internally for the specific hardware target. It is possible to export compiled model using <a href="#import-export">Import/Export</a> functionality to use it later, but in the general case, there is no guarantee that a model compiled and exported for GNA 2.0 runs on GNA 3.0, or vice versa.

@sphinxdirective

.. csv-table:: Interoperability of compile target and hardware target
   :header: "Hardware", "Compile target 2.0", "Compile target 3.0"

   "GNA 2.0", "Supported", "Not supported (incompatible layers emulated on CPU)"
   "GNA 3.0", "Partially supported", "Supported"

@endsphinxdirective

> **NOTE**: In most cases, networks compiled for GNA 2.0 runs as expected on GNA 3.0, although the performance may be worse compared to the case when a network is compiled specifically for the latter.  The exception is networks with convolutions with the number of filters greater than 8192 (see the <a href="#models-and-operations-limitations">Models and Operations Limitations</a> section).

For optimal work with POT quantized models which includes 2D convolutions on GNA 3.0 hardware, the <a href="#support-for-2d-convolutions-using-pot">following requirements</a> should be satisfied.

Choose a compile target depending on the priority: cross-platform execution, performance, memory, or power optimization..

Use the following properties to check interoperability in your application: `ov::intel_gna::execution_target` and `ov::intel_gna::compile_target`

[Speech C++ Sample](@ref openvino_inference_engine_samples_speech_sample_README) can be used for experiments (see `-exec_target` and `-compile_target` command line options).

## Software emulation mode

On platforms without GNA hardware support plugin chooses software emulation mode by default. It means, model runs even if you do not have GNA HW within your platform.
GNA plugin enables you to switch the execution between software emulation mode and hardware execution mode after the model is loaded.
For details, see description of the `ov::intel_gna::execution_mode` property.

## Recovery from Interruption by High-Priority Windows Audio Processes\*

GNA is designed for real-time workloads such as noise reduction.
For such workloads, processing should be time constrained, otherwise extra delays may cause undesired effects such as
*audio glitches*. To make sure that processing can satisfy real-time requirements, the GNA driver provides a Quality of Service
(QoS) mechanism, which interrupts requests that might cause high-priority Windows audio processes to miss
the schedule, thereby causing long running GNA tasks to terminate early.

To prepare the applications correctly, use Automatic QoS Feature described below.

### Automatic QoS Feature on Windows*

Starting with 2021.4.1 release of OpenVINO and 03.00.00.1363 version of Windows* GNA driver, a new execution mode `ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK` is introduced
to assure that workloads satisfy real-time execution. In this mode, the GNA driver automatically falls back on CPU for a particular infer request
if the HW queue is not empty, so there is no need for explicitly switching between GNA and CPU.

@sphinxdirective
.. tab:: C++

   .. doxygensnippet:: docs/snippets/gna/configure.cpp
      :language: cpp
      :fragment: [include]

   .. doxygensnippet:: docs/snippets/gna/configure.cpp
      :language: cpp
      :fragment: [ov_gna_exec_mode_hw_with_sw_fback]

.. tab:: Python

   .. doxygensnippet:: docs/snippets/gna/configure.py
      :language: python
      :fragment: [import]

   .. doxygensnippet:: docs/snippets/gna/configure.py
      :language: python
      :fragment: [ov_gna_exec_mode_hw_with_sw_fback]

@endsphinxdirective

> **NOTE**: Due to the "first come - first served" nature of GNA driver and the QoS feature, this mode may lead to increased CPU consumption
if there are several clients using GNA simultaneously.
Even a lightweight competing infer request which has not been cleared at the time when the user's GNA client process makes its request,
can cause the user's request to be executed on CPU, thereby unnecessarily increasing CPU utilization and power.

## Supported inference data types

Intel® GNA essentially operates in the low-precision mode which represents a mix of 8-bit (`i8`), 16-bit (`i16`), and 32-bit (`i32`) integer computations.

GNA plugin users are encouraged to use the [Post-Training Optimization Tool](@ref pot_README) to get a model with quantization hints based on statistics for the provided dataset.

Unlike other plugins supporting low-precision execution, the GNA plugin can calculate quantization factors at the model loading time, so you can run a model without calibration. However, this mode may not provide satisfactory accuracy because the internal quantization algorithm is based on heuristics which may or may not be efficient, depending on the model and dynamic range of input data and this mode is going to be deprecated soon.

GNA plugin supports the following data types as inference precision of internal primitives
* Quantized data types:
  - i16
  - i8

[Hello Query Device C++ Sample](@ref openvino_inference_engine_samples_hello_query_device_README) can be used to print out supported data types for all detected devices.

[POT API Usage sample for GNA](@ref pot_sample_speech_README) demonstrates how a model can be quantized for GNA using POT API in 2 modes:
* Accuracy (i16 weights)
* Performance (i8 weights)

For POT quantized model `ov::hint::inference_precision` property has no effect except cases described in <a href="#support-for-2d-convolutions-using-pot">Support for 2D Convolutions using POT</a>.

## Supported features

### Models caching
Cache for GNA plugin may be enabled via common OpenVINO `ov::cache_dir` property due to import/export functionality support (see below).

See [Model caching overview page](@ref openvino_docs_IE_DG_Model_caching_overview) for more details.

### Import/Export

The GNA plugin supports import/export capability which helps to significantly decrease first inference time. The model compile target is the same as the execution target by default. The default value for the execution target corresponds to available hardware, or latest hardware version supported by the plugin (i.e., GNA 3.0) if there is no GNA HW in the system.

If you are willing to export a model for a specific version of GNA HW, please use the `ov::intel_gna::compile_target` property and then export the model:

@sphinxdirective
.. tab:: C++

   .. doxygensnippet:: docs/snippets/gna/import_export.cpp
      :language: cpp
      :fragment: [ov_gna_export]

.. tab:: Python

   .. doxygensnippet:: docs/snippets/gna/import_export.py
      :language: python
      :fragment: [ov_gna_export]

@endsphinxdirective

Import model:

@sphinxdirective
.. tab:: C++

   .. doxygensnippet:: docs/snippets/gna/import_export.cpp
      :language: cpp
      :fragment: [ov_gna_import]

.. tab:: Python

   .. doxygensnippet:: docs/snippets/gna/import_export.py
      :language: python
      :fragment: [ov_gna_import]

@endsphinxdirective

[Compile Tool](@ref openvino_inference_engine_tools_compile_tool_README) or [Speech C++ Sample](@ref openvino_inference_engine_samples_speech_sample_README) can be used to compile model.

### Stateful models
GNA plugin natively supports stateful models.

Please refer to [Stateful models] (@ref openvino_docs_IE_DG_network_state_intro) for more details about such models.

> **NOTE**: Typically, GNA is used in streaming scenarios, when minimizing the latency is important. Taking into account that POT does not support the `TensorIterator` operation, the recommendation is to use the `--transform` option of the Model Optimizer to apply `LowLatency2` transformation when converting an original model.

### Profiling
The GNA plugin allows to turn on profiling using the `ov::enable_profiling` property.
With the following methods, you can collect profiling information that provides various performance data about execution on GNA:

@sphinxdirective
.. tab:: C++

   ``ov::InferRequest::get_profiling_info``

.. tab:: Python

   ``openvino.runtime.InferRequest.get_profiling_info``

@endsphinxdirective

The current GNA implementation calculates counters for the whole utterance scoring and does not provide per-layer information. The API enables you to retrieve counter units in cycles, you can convert cycles to seconds as follows:

```
seconds = cycles / frequency
```

Refer to the table below to learn about the frequency of Intel® GNA inside a particular processor:

@sphinxdirective

.. csv-table:: Frequency of Intel® GNA inside a particular processor
   :header: "Processor", "Frequency of Intel® GNA, MHz"

   "Intel® Core™ processors", 400
   "Intel® processors formerly codenamed Elkhart Lake", 200
   "Intel® processors formerly codenamed Gemini Lake", 200

@endsphinxdirective

Performance counters provided for the time being:

* Inference request performance results
	* Number of total cycles spent on scoring in hardware including compute and memory stall cycles
	* Number of stall cycles spent in hardware

##  Supported properties
The plugin supports the properties listed below.

### Read-write properties
The following parameters must be set before model compilation in order to take effect or passed as additional argument to `ov::Core::compile_model()`:

- ov::cache_dir
- ov::enable_profiling
- ov::hint::inference_precision
- ov::hint::num_requests
- ov::intel_gna::compile_target
- ov::intel_gna::firmware_model_image_path
- ov::intel_gna::execution_target
- ov::intel_gna::pwl_design_algorithm
- ov::intel_gna::pwl_max_error_percent
- ov::intel_gna::scale_factors_per_input

These parameters can be changed after model compilation `ov::CompiledModel::set_property`:
- ov::hint::performance_mode
- ov::intel_gna::execution_mode
- ov::log::level

### Read-only properties
- ov::available_devices
- ov::device::capabilities
- ov::device::full_name
- ov::intel_gna::library_full_version
- ov::optimal_number_of_infer_requests
- ov::range_for_async_infer_requests
- ov::supported_properties

## Limitations

### Models and Operations Limitations

Because of specifics of hardware architecture, Intel® GNA supports a limited set of operations, their kinds and combinations.
For example, you should not expect the GNA Plugin to be able to run computer vision models, except those specifically adapted for the GNA Plugin, because the plugin does not fully support 2D convolutions.

Limitations include:

- Only 1D convolutions are natively supported on the HW prior to GNA 3.0; 2D convolutions have specific limitations (see the table below).
- The number of output channels for convolutions must be a multiple of 4.
- The maximum number of filters is 65532 for GNA 2.0 and 8192 for GNA 3.0.
- Transpose layer support is limited to the cases where no data reordering is needed or when reordering is happening for two dimensions, at least one of which is not greater than 8.
- Splits and concatenations are supported for continuous portions of memory (e.g., split of 1,2,3,4 to 1,1,3,4 and 1,1,3,4 or concats of 1,2,3,4 and 1,2,3,5 to 2,2,3,4).
- For Multiply, Add and Subtract layers, auto broadcasting is only supported for constant inputs.

#### Support for 2D Convolutions

The Intel® GNA 1.0 and 2.0 hardware natively supports only 1D convolutions. However, 2D convolutions can be mapped to 1D when a convolution kernel moves in a single direction.

Initially, a limited subset of Intel® GNA 3.0 features are added to the previous feature set including the following:

* **2D VALID Convolution With Small 2D Kernels:**  Two-dimensional convolutions with the following kernel dimensions [H,W] are supported: [1,1], [2,2], [3,3], [2,1], [3,1], [4,1], [5,1], [6,1], [7,1], [1,2], or [1,3]. Input tensor dimensions are limited to [1,8,16,16] <= [N,C,H,W] <= [1,120,384,240]. Up to 384 channels C may be used with a subset of kernel sizes (see table below).  Up to 256 kernels (output channels) are supported. Pooling is limited to pool shapes of [1,1], [2,2], or [3,3]. Not all combinations of kernel shape and input tensor shape are supported (see the tables below for exact limitations).

The tables below show that the exact limitation on the input tensor width W depends on the number of input channels C (indicated as Ci below) and the kernel shape.  There is much more freedom to choose the input tensor height and number of output channels.

The following tables provide a more explicit representation of the Intel(R) GNA 3.0 2D convolution operations initially supported. The limits depend strongly on number of input tensor channels (Ci) and the input tensor width (W). Other factors are kernel height (KH), kernel width (KW), pool height (PH), pool width (PW), horizontal pool step (SH), and vertical pool step (PW). For example, the first table shows that for a 3x3 kernel with max pooling, only square pools are supported, and W is limited to 87 when there are 64 input channels.

@sphinxdirective

:download:`Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters (Input and Kernel Precision: i16) <../../../docs/OV_Runtime_UG/supported_plugins/files/GNA_Maximum_Input_Tensor_Widths_i16.csv>`

:download:`Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters (Input and Kernel Precision: i8) <../../../docs/OV_Runtime_UG/supported_plugins/files/GNA_Maximum_Input_Tensor_Widths_i8.csv>`

@endsphinxdirective

> **NOTE**: The above limitations only apply to the new hardware 2D convolution operation. When possible, the Intel® GNA plugin graph compiler flattens 2D convolutions so that the second generation Intel® GNA 1D convolution operations (without these limitations) may be used. The plugin will also flatten 2D convolutions regardless of the sizes if GNA 2.0 compilation target is selected (see below).

#### Support for 2D Convolutions using POT

For POT to successfully work with the models including GNA3.0 2D convolutions, the following requirements must be met:
* All convolution parameters are natively supported by HW (see tables above)
* The runtime precision is explicitly set by the `ov::hint::inference_precision` property as `i8` for the models produced by the `performance mode` of POT, and as `i16` for the models produced by the `accuracy mode` of POT.

### Batch Size Limitation

Intel® GNA plugin supports the processing of context-windowed speech frames in batches of 1-8 frames.

Please refer to [Layout API overview](@ref openvino_docs_OV_Runtime_UG_Layout_Overview) to determine batch dimension.

To set layout of model inputs in runtime use [Preprocessing API](@ref openvino_docs_OV_Runtime_UG_Preprocessing_Overview):

@sphinxdirective
.. tab:: C++

   .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
      :language: cpp
      :fragment: [include]

   .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
      :language: cpp
      :fragment: [ov_gna_set_nc_layout]

.. tab:: Python

   .. doxygensnippet:: docs/snippets/gna/set_batch.py
      :language: python
      :fragment: [import]

   .. doxygensnippet:: docs/snippets/gna/set_batch.py
      :language: python
      :fragment: [ov_gna_set_nc_layout]

@endsphinxdirective

then set batch size:

@sphinxdirective
.. tab:: C++

   .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
      :language: cpp
      :fragment: [ov_gna_set_batch_size]

.. tab:: Python

   .. doxygensnippet:: docs/snippets/gna/set_batch.py
      :language: python
      :fragment: [ov_gna_set_batch_size]

@endsphinxdirective

Increasing batch size only improves efficiency of `MatMul` layers.

> **NOTE**: For models with `Convolution`, `LSTMCell`, or `ReadValue`/`Assign` operations, the only supported batch size is 1.

### Compatibility with Heterogeneous mode

[Heterogeneous execution](@ref openvino_docs_OV_UG_Hetero_execution) is currently not supported by GNA plugin.

## See Also

* [Supported Devices](Supported_Devices.md)
* [Converting Model](../../MO_DG/prepare_model/convert_model/Converting_Model.md)
* [Convert model from Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)