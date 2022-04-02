# OpenVINO™ Python API exclusives {#openvino_docs_OV_UG_Python_API_exclusives}

OpenVINO™ Runtime Python API is exposing additional features and helpers to elevate user experience. Main goal of Python API is to provide user-friendly and simple, still powerful, tool for Python users.

## Easier model compilation 

`CompiledModel` can be easily created with the helper method. It hides `Core` creation and applies `AUTO` device by default.

@snippet docs/snippets/ov_python_exclusives.py auto_compilation

## Model/CompiledModel inputs and outputs

Besides functions aligned to C++ API, some of them have their Pythonic counterparts or extensions. For example, `Model` and `CompiledModel` inputs/outputs can be accessed via properties.

@snippet docs/snippets/ov_python_exclusives.py properties_example

Refer to Python API documentation on which helper functions or properties are available for different classes.

## Working with Tensor

Python API allows passing data as tensors. `Tensor` object holds a copy of the data from the given array. `dtype` of numpy arrays is converted to OpenVINO™ types automatically.

@snippet docs/snippets/ov_python_exclusives.py tensor_basics

### Shared memory mode

`Tensor` objects can share the memory with numpy arrays. By specifing `shared_memory` argument, a `Tensor` object does not perform copy of data and has access to the memory of the numpy array.

@snippet docs/snippets/ov_python_exclusives.py tensor_shared_mode

## Running inference

Python API supports extra calling methods to synchronous and asynchronous modes for inference.

All infer methods allow users to pass data as popular numpy arrays, gathered in either Python dicts or lists.

@snippet docs/snippets/ov_python_exclusives.py passing_numpy_array

Results from inference can be obtained in various ways:

@snippet docs/snippets/ov_python_exclusives.py getting_results

### Synchronous mode - extended

Python API provides different synchronous calls to infer model, which block the application execution. Additionally these calls return results of inference:

@snippet docs/snippets/ov_python_exclusives.py sync_infer

### AsyncInferQueue

Asynchronous mode pipelines can be supported with wrapper class called `AsyncInferQueue`. This class automatically spawns pool of `InferRequest` objects (also called "jobs") and provides synchronization mechanisms to control flow of the pipeline.

Each job is distinguishable by unique `id`, which is in the range from 0 up to number of jobs specified in `AsyncInferQueue` constructor.

Function call `start_async` is not required to be synchronized, it waits for any available job if queue is busy/overloaded. Every `AsyncInferQueue` code block should end with `wait_all` function. It provides "global" synchronization of all jobs in the pool and ensure that access to them is safe.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue

#### Acquire results from requests

After the call to `wait_all`, jobs and their data can be safely accessed. Acquring of a specific job with `[id]` returns `InferRequest` object, which results in seamless retrieval of the output data.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue_access

#### Setting callbacks

Another feature of `AsyncInferQueue` is ability of setting callbacks. When callback is set, any job that ends inference, calls upon Python function. Callback function must have two arguments. First is the request that calls the callback, it provides `InferRequest` API. Second one being called "userdata", provides possibility of passing runtime values, which can be of any Python type and later used inside callback function.

The callback of `AsyncInferQueue` is uniform for every job. When executed, GIL is acquired to ensure safety of data manipulation inside the function.

@snippet docs/snippets/ov_python_exclusives.py asyncinferqueue_set_callback

### Working with u1, u4 and i4 element types

Since openvino supports low precision element types there are few ways how to handle them in python.
To create an input tensor with such element types you may need to pack your data in new numpy array which byte size matches original input size:
@snippet docs/snippets/ov_python_exclusives.py packing_data

To extract low precision values from tensor into numpy array you can use next helper:
@snippet docs/snippets/ov_python_exclusives.py unpacking

### Releasing the GIL

Some functions in Python API release the Global Lock Interpreter (GIL) while running work-intensive code. It can help you to achieve more parallelism in your application using Python threads. For more information about GIL please refer to the Python documentation.

@snippet docs/snippets/ov_python_exclusives.py releasing_gil

> **NOTE**: While GIL is released functions can still modify and/or operate on Python objects in C++, thus there is no reference counting. User is responsible for thread safety if sharing of these objects with other thread occurs. It can affects your code only if multiple threads are spawned in Python.:

#### List of functions that release the GIL
- openvino.runtime.AsyncInferQueue.start_async
- openvino.runtime.AsyncInferQueue.is_ready
- openvino.runtime.AsyncInferQueue.wait_all
- openvino.runtime.AsyncInferQueue.get_idle_request_id
- openvino.runtime.CompiledModel.create_infer_request
- openvino.runtime.CompiledModel.infer_new_request
- openvino.runtime.CompiledModel.__call__
- openvino.runtime.CompiledModel.export
- openvino.runtime.CompiledModel.get_runtime_model
- openvino.runtime.Core.compile_model
- openvino.runtime.Core.read_model
- openvino.runtime.Core.import_model
- openvino.runtime.Core.query_model
- openvino.runtime.Core.get_available_devices
- openvino.runtime.InferRequest.infer
- openvino.runtime.InferRequest.start_async
- openvino.runtime.InferRequest.wait
- openvino.runtime.InferRequest.wait_for
- openvino.runtime.InferRequest.get_profiling_info
- openvino.runtime.InferRequest.query_state
- openvino.runtime.Model.reshape
- openvino.preprocess.PrePostProcessor.build
