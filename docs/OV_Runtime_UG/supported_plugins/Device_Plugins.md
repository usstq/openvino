# Working with devices {#openvino_docs_OV_UG_Working_with_devices}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_query_api
   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_IE_DG_supported_plugins_VPU
   openvino_docs_OV_UG_supported_plugins_GNA

@endsphinxdirective

The OpenVINO Runtime provides capabilities to infer deep learning models on the following device types with corresponding plugins:

| Plugin | Device types                                                                                                                                                |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[CPU](CPU.md)              |Intel&reg; Xeon&reg; with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel&reg; Core&trade; Processors with Intel&reg; AVX2, Intel&reg; Atom&reg; Processors with Intel® Streaming SIMD Extensions (Intel® SSE) |
|[GPU](GPU.md)            |Intel® Graphics, including Intel® HD Graphics, Intel® UHD Graphics, Intel® Iris® Graphics, Intel® Xe Graphics, Intel® Xe MAX Graphics |
|[VPUs](VPU.md)            |Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs                                                                                           |
|[GNA](GNA.md)              |[Intel® Speech Enabling Developer Kit](https://www.intel.com/content/www/us/en/support/articles/000026156/boards-and-kits/smart-home.html); [Amazon Alexa\* Premium Far-Field Developer Kit](https://developer.amazon.com/en-US/alexa/alexa-voice-service/dev-kits/amazon-premium-voice); [Intel® Pentium® Silver Processors N5xxx, J5xxx and Intel® Celeron® Processors N4xxx, J4xxx (formerly codenamed Gemini Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/83915/gemini-lake.html): [Intel® Pentium® Silver J5005 Processor](https://ark.intel.com/content/www/us/en/ark/products/128984/intel-pentium-silver-j5005-processor-4m-cache-up-to-2-80-ghz.html), [Intel® Pentium® Silver N5000 Processor](https://ark.intel.com/content/www/us/en/ark/products/128990/intel-pentium-silver-n5000-processor-4m-cache-up-to-2-70-ghz.html), [Intel® Celeron® J4005 Processor](https://ark.intel.com/content/www/us/en/ark/products/128992/intel-celeron-j4005-processor-4m-cache-up-to-2-70-ghz.html), [Intel® Celeron® J4105 Processor](https://ark.intel.com/content/www/us/en/ark/products/128989/intel-celeron-j4105-processor-4m-cache-up-to-2-50-ghz.html), [Intel® Celeron® J4125 Processor](https://ark.intel.com/content/www/us/en/ark/products/197305/intel-celeron-processor-j4125-4m-cache-up-to-2-70-ghz.html), [Intel® Celeron® Processor N4100](https://ark.intel.com/content/www/us/en/ark/products/128983/intel-celeron-processor-n4100-4m-cache-up-to-2-40-ghz.html), [Intel® Celeron® Processor N4000](https://ark.intel.com/content/www/us/en/ark/products/128988/intel-celeron-processor-n4000-4m-cache-up-to-2-60-ghz.html); [Intel® Pentium® Processors N6xxx, J6xxx, Intel® Celeron® Processors N6xxx, J6xxx and Intel Atom® x6xxxxx (formerly codenamed Elkhart Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/128825/products-formerly-elkhart-lake.html); [Intel® Core™ Processors (formerly codenamed Cannon Lake)](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html); [10th Generation Intel® Core™ Processors (formerly codenamed Ice Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/ice-lake.html): [Intel® Core™ i7-1065G7 Processor](https://ark.intel.com/content/www/us/en/ark/products/196597/intel-core-i71065g7-processor-8m-cache-up-to-3-90-ghz.html), [Intel® Core™ i7-1060G7 Processor](https://ark.intel.com/content/www/us/en/ark/products/197120/intel-core-i71060g7-processor-8m-cache-up-to-3-80-ghz.html), [Intel® Core™ i5-1035G4 Processor](https://ark.intel.com/content/www/us/en/ark/products/196591/intel-core-i51035g4-processor-6m-cache-up-to-3-70-ghz.html), [Intel® Core™ i5-1035G7 Processor](https://ark.intel.com/content/www/us/en/ark/products/196592/intel-core-i51035g7-processor-6m-cache-up-to-3-70-ghz.html), [Intel® Core™ i5-1035G1 Processor](https://ark.intel.com/content/www/us/en/ark/products/196603/intel-core-i51035g1-processor-6m-cache-up-to-3-60-ghz.html), [Intel® Core™ i5-1030G7 Processor](https://ark.intel.com/content/www/us/en/ark/products/197119/intel-core-i51030g7-processor-6m-cache-up-to-3-50-ghz.html), [Intel® Core™ i5-1030G4 Processor](https://ark.intel.com/content/www/us/en/ark/products/197121/intel-core-i51030g4-processor-6m-cache-up-to-3-50-ghz.html), [Intel® Core™ i3-1005G1 Processor](https://ark.intel.com/content/www/us/en/ark/products/196588/intel-core-i31005g1-processor-4m-cache-up-to-3-40-ghz.html), [Intel® Core™ i3-1000G1 Processor](https://ark.intel.com/content/www/us/en/ark/products/197122/intel-core-i31000g1-processor-4m-cache-up-to-3-20-ghz.html), [Intel® Core™ i3-1000G4 Processor](https://ark.intel.com/content/www/us/en/ark/products/197123/intel-core-i31000g4-processor-4m-cache-up-to-3-20-ghz.html); [11th Generation Intel® Core™ Processors (formerly codenamed Tiger Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/tiger-lake.html); [12th Generation Intel® Core™ Processors (formerly codenamed Alder Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/147470/products-formerly-alder-lake.html)|

OpenVINO runtime also has several execution capabilities which work on top of other devices:

| Capability                               | Description                                                                                                                                                 |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[Multi-Device execution](../multi_device.md) |Multi-Device enables simultaneous inference of the same model on several devices in parallel    |
|[Auto-Device selection](../auto_device_selection.md) |Auto-Device selection enables selecting Intel&reg; device for inference automatically |
|[Heterogeneous execution](../hetero_execution.md) |Heterogeneous execution enables automatic inference splitting between several devices (for example if a device doesn't [support certain operation](#supported-layers))|
|[Automatic Batching](../automatic_batching.md) | Auto-Batching plugin enables the batching (on top of the specified device)  that is completely transparent to the application |

Devices similar to the ones we have used for benchmarking can be accessed using [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/), a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. [Learn more](https://devcloud.intel.com/edge/get_started/devcloud/) or [Register here](https://inteliot.force.com/DevcloudForEdge/s/).


## Features support matrix
The table below demonstrates support of key features by OpenVINO device plugins.

| Capability | [CPU](CPU.md) | [GPU](GPU.md) | [GNA](GNA.md) | [VPU](VPU.md) |
| ---------- | --- | --- | --- | --- |
| [Heterogeneous execution](../hetero_execution.md)| Yes | Yes | No | ? |
| [Multi-device execution](../multi_device.md) | Yes | Yes | Partial | ? |
| [Automatic batching](../automatic_batching.md) | No | Yes | No | ? |
| [Multi-stream execution](@ref openvino_docs_optimization_guide_dldt_optimization_guide) | Yes | Yes | No | ? |
| [Models caching](../Model_caching_overview.md) | Yes | Partial | Yes | ? |
| [Dynamic shapes](../ov_dynamic_shapes.md) | Yes | Partial | No | ? |
| Import/Export | Yes | No | Yes | ? |
| [Preprocessing acceleration](../preprocessing_overview.md) | Yes | Yes | No | ? |
| [Stateful models](../network_state_intro.md) | Yes | No | Yes | ? |
| [Extensibility](@ref openvino_docs_Extensibility_UG_Intro) | Yes | Yes | No | ? |

For more details on plugin specific feature limitation, see corresponding plugin pages.
