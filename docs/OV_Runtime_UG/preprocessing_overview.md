# Overview of Preprocessing API {#openvino_docs_OV_Runtime_UG_Preprocessing_Overview}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_Runtime_UG_Preprocessing_Details
   openvino_docs_OV_Runtime_UG_Layout_Overview
   openvino_docs_OV_Runtime_UG_Preprocess_Usecase_save

@endsphinxdirective

## Introduction

When your input data don't perfectly fit to Neural Network model input tensor - this means that additional operations/steps are needed to transform your data to format expected by model. These operations are known as "preprocessing".

### Example
Consider the following standard example: deep learning model expects input with shape `{1, 3, 224, 224}`, `FP32` precision, `RGB` color channels order, and requires data normalization (subtract mean and divide by scale factor). But you have just a `640x480` `BGR` image (data is `{480, 640, 3}`). This means that we need some operations which will:
 - Convert U8 buffer to FP32
 - Transform to `planar` format: from `{1, 480, 640, 3}` to `{1, 3, 480, 640}`
 - Resize image from 640x480 to 224x224
 - Make `BGR->RGB` conversion as model expects `RGB`
 - For each pixel, subtract mean values and divide by scale factor


![](img/preprocess_not_fit.png)


Even though all these steps can be relatively easy implemented manually in application's code before actual inference, it is possible to do it with Preprocessing API. Reasons to use this API are:
 - Preprocessing API is easy to use
 - Preprocessing steps will be integrated into execution graph and will be performed on selected device (CPU/GPU/VPU/etc.) rather than always being executed on CPU. This will improve selected device utilization which is always good.

## Preprocessing API

Intuitively, Preprocessing API consists of the following parts:
 1. 	**Tensor:** Declare user's data format, like shape, [layout](./layout_overview.md), precision, color format of actual user's data
 2. 	**Steps:** Describe sequence of preprocessing steps which need to be applied to user's data
 3. 	**Model:** Specify Model's data format. Usually, precision and shape are already known for model, only additional information, like [layout](./layout_overview.md) can be specified

> **Note:** All model's graph modification shall be performed after model is read from disk and **before** it is being loaded on actual device.

### PrePostProcessor object

`ov::preprocess::PrePostProcessor` class allows specifying preprocessing and postprocessing steps for model read from disk.

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:create]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:create]

@endsphinxdirective

### Declare user's data format

To address particular input of model/preprocessor, use `ov::preprocess::PrePostProcessor::input(input_name)` method

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:tensor]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:tensor]

@endsphinxdirective


Here we've specified all information about user's input:
 - Precision is U8 (unsigned 8-bit integer)
 - Data represents tensor with {1,480,640,3} shape
 - [Layout](./layout_overview.md) is "NHWC". It means that 'height=480, width=640, channels=3'
 - Color format is `BGR`

### Declare model's layout

Model's input already has information about precision and shape. Preprocessing API is not intended to modify this. The only thing that may be specified is input's data [layout](./layout_overview.md)

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:model]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:model]

@endsphinxdirective


Now, if model's input has `{1,3,224,224}` shape, preprocessing will be able to identify that model's `height=224`, `width=224`, `channels=3`. Height/width information is necessary for 'resize', and `channels` is needed for mean/scale normalization

### Preprocessing steps

Now we can define sequence of preprocessing steps:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:steps]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:steps]

@endsphinxdirective

Here:
 - Convert U8 to FP32 precision
 - Convert current color format (BGR) to RGB
 - Resize to model's height/width. **Note** that if model accepts dynamic size, e.g. {?, 3, ?, ?}, `resize` will not know how to resize the picture, so in this case you should specify target height/width on this step. See also <code>ov::preprocess::PreProcessSteps::resize()</code>
 - Subtract mean from each channel. On this step, color format is RGB already, so `100.5` will be subtracted from each Red component, and `101.5` will be subtracted from `Blue` one.
 - Divide each pixel data to appropriate scale value. In this example each `Red` component will be divided by 50, `Green` by 51, `Blue` by 52 respectively
 - **Note:** last `convert_layout` step is commented out as it is not necessary to specify last layout conversion. PrePostProcessor will do such conversion automatically

### Integrate steps into model

We've finished with preprocessing steps declaration, now it is time to build it. For debugging purposes it is possible to print `PrePostProcessor` configuration on screen:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: [ov:preprocess:build]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: python
         :fragment: [ov:preprocess:build]

@endsphinxdirective


After this, `model` will accept U8 input with `{1, 480, 640, 3}` shape, with `BGR` channels order. All conversion steps will be integrated into execution graph. Now you can load model on device and pass your image to model as is, without any data manipulation on application's side


## See Also

* [Preprocessing Details](./preprocessing_details.md)
* [Layout API overview](./layout_overview.md)
* <code>ov::preprocess::PrePostProcessor</code> C++ class documentation
