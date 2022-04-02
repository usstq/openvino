# Performing inference with OpenVINO Runtime {#openvino_docs_OV_UG_OV_Runtime_User_Guide}

@sphinxdirective

.. _deep learning openvino runtime:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Integrate_OV_with_your_application
   openvino_docs_OV_UG_ShapeInference
   openvino_docs_OV_UG_Working_with_devices
   openvino_docs_OV_UG_Preprocessing_Overview
   openvino_docs_OV_UG_DynamicShapes
   openvino_docs_OV_UG_supported_plugins_AUTO
   openvino_docs_OV_UG_Running_on_multiple_devices
   openvino_docs_OV_UG_Hetero_execution
   openvino_docs_OV_UG_Performance_Hints
   openvino_docs_OV_UG_Automatic_Batching
   openvino_docs_OV_UG_network_state_intro
   
@endsphinxdirective

## Introduction
OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the OpenVINO Runtime API to read an Intermediate Representation (IR), ONNX, or PaddlePaddle model and execute it on preferred devices.

OpenVINO Runtime uses a plugin architecture. Its plugins are software components that contain complete implementation for inference on a particular Intel® hardware device: CPU, GPU, VPU, etc. Each plugin implements the unified API and provides additional hardware-specific APIs, for configuring devices, or API interoperability between OpenVINO Runtime and underlying plugin backend.
 
The scheme below illustrates the typical workflow for deploying a trained deep learning model: 

<!-- TODO: need to update the picture below with PDPD files -->
![](img/BASIC_FLOW_IE_C.svg)


## Video

@sphinxdirective

.. list-table::

   * - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen height="315" width="100%"
           src="https://www.youtube.com/embed/e6R13V8nbak">
           </iframe>
   * - **OpenVINO Runtime Concept**. Duration: 3:43
     
@endsphinxdirective
