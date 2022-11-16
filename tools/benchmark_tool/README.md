# Benchmark Python Tool {#openvino_inference_engine_tools_benchmark_tool_README}

This page demonstrates how to use the Benchmark Python Tool to estimate deep learning inference performance on supported devices.

> **NOTE**: This page describes usage of the Python implementation of the Benchmark Tool. For the C++ implementation, refer to the [Benchmark C++ Tool](../../samples/cpp/benchmark_app/README.md) page. The Python version is recommended for benchmarking models that will be used in Python applications, and the C++ version is recommended for benchmarking models that will be used in C++ applications. Both tools have a similar command interface and backend.

## Basic Usage

The Python benchmark_app is automatically installed when you install OpenVINO Developer Tools using [PyPI](../../docs/install_guides/installing-openvino-pip.md). Before running `benchmark_app`, make sure the `openvino_env` virtual environment is activated, and navigate to the directory where your model is located.

The benchmarking application works with models in the OpenVINO IR (`model.xml` and `model.bin`) and ONNX (`model.onnx`) formats. Make sure to [convert your models](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) if necessary. 

To run benchmarking with default options on a model, use the following command:

```
benchmark_app -m model.xml
```

By default, the application will load the specified model onto the CPU and perform inferencing on batches of randomly-generated data inputs for 60 seconds. As it loads, it prints information about benchmark parameters. When benchmarking is completed, it reports the minimum, average, and maximum inferencing latency and average the throughput.

You may be able to improve benchmark results beyond the default configuration by configuring some of the execution parameters for your model. For example, you can use "throughput" or "latency" performance hints to optimize the runtime for higher FPS or reduced inferencing time. Read on to learn more about the configuration options available with benchmark_app.

## Configuration Options
The benchmark app provides various options for configuring execution parameters. This section covers key configuration options for easily tuning benchmarking to achieve better performance on your device. A list of all configuration options is given in the [Advanced Usage](#advanced-usage) section.

### Performance hints: latency and throughput
The benchmark app allows users to provide high-level "performance hints" for setting latency-focused or throughput-focused inference modes. This hint causes the runtime to automatically adjust runtime parameters, such as the number of processing streams and inference batch size, to prioritize for reduced latency or high throughput.

The performance hints do not require any device-specific settings and they are completely portable between devices. Parameters are automatically configured based on whichever device is being used. This allows users to easily port applications between hardware targets without having to re-determine the best runtime parameters for the new device.

If not specified, throughput is used as the default. To set the hint explicitly, use `-hint latency` or `-hint throughput` when running benchmark_app:

```
benchmark_app -m model.xml -hint latency
benchmark_app -m model.xml -hint throughput
```

#### Latency
Latency is the amount of time it takes to process a single inference request. In applications where data needs to be inferenced and acted on as quickly as possible (such as autonomous driving), low latency is desirable. For conventional devices, lower latency is achieved by reducing the amount of parallel processing streams so the system can utilize as many resources as possible to quickly calculate each inference request. However, advanced devices like multi-socket CPUs and modern GPUs are capable of running multiple inference requests while delivering the same latency.

When benchmark_app is run with `-hint latency`, it determines the optimal number of parallel inference requests for minimizing latency while still maximizing the parallelization capabilities of the hardware. It automatically sets the number of processing streams and inference batch size to achieve the best latency.

#### Throughput
Throughput is the amount of data an inferencing pipeline can process at once, and it is usually measured in frames per second (FPS) or inferences per second. In applications where large amounts of data needs to be inferenced simultaneously (such as multi-camera video streams), high throughput is needed. To achieve high throughput, the runtime focuses on fully saturating the device with enough data to process. It utilizes as much memory and as many parallel streams as possible to maximize the amount of data that can be processed simultaneously.

When benchmark_app is run with `-hint throughput`, it maximizes the number of parallel inference requests to utilize all the threads available on the device. On GPU, it automatically sets the inference batch size to fill up the GPU memory available.

For more information on performance hints, see the [High-level Performance Hints](../../docs/OV_Runtime_UG/performance_hints.md) page. For more details on optimal runtime configurations and how they are automatically determined using performance hints, see [Runtime Inference Optimizations](../../docs/optimization_guide/dldt_deployment_optimization_guide.md).


### Device
To set which device benchmarking runs on, use the `-d <device>` argument. This will tell benchmark_app to run benchmarking on that specific device. The benchmark app supports "CPU", "GPU", and "MYRIAD" (also known as [VPU](../../docs/OV_Runtime_UG/supported_plugins/VPU.md)) devices. In order to use the GPU or VPU, the system must have the appropriate drivers installed. If no device is specified, benchmark_app will default to using CPU.

For example, to run benchmarking on GPU, use:

```
benchmark_app -m model.xml -d GPU
```

You may also specify "AUTO" as the device, in which case the benchmark_app will automatically select the best device for benchmarking and support it with the CPU at the model loading stage. This may result in increased performance, thus, should be used purposefully. For more information, see the [Automatic device selection](../../docs/OV_Runtime_UG/auto_device_selection.md) page.

(Note: If the latency or throughput hint is set, it will automatically configure streams and batch sizes for optimal performance based on the specified device.)

### Number of iterations
By default, the benchmarking app will run for a predefined duration, repeatedly performing inferencing with the model and measuring the resulting inference speed. There are several options for setting the number of inference iterations:

* Explicitly specify the number of iterations the model runs using the `-niter <number_of_iterations>` option
* Set how much time the app runs for using the `-t <seconds>` option
* Set both of them (execution will continue until both conditions are met)
* If neither -niter nor -t are specified, the app will run for a predefined duration that depends on the device

The more iterations a model runs, the better the statistics will be for determing average latency and throughput.

### Inputs
The benchmark tool runs benchmarking on user-provided input images in `.jpg`, `.bmp`, or `.png` format. Use `-i <PATH_TO_INPUT>` to specify the path to an image, or folder of images. For example, to run benchmarking on an image named `test1.jpg`, use:

```
./benchmark_app -m model.xml -i test1.jpg
```

The tool will repeatedly loop through the provided inputs and run inferencing on them for the specified amount of time or number of iterations. If the `-i` flag is not used, the tool will automatically generate random data to fit the input shape of the model. 

### Examples
For more usage examples (and step-by-step instructions on how to set up a model for benchmarking), see the [Examples of Running the Tool](#examples-of-running-the-tool) section.

## Advanced Usage

> **NOTE**: By default, OpenVINO samples, tools and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channel order in the sample or demo application or reconvert your model using the Model Optimizer tool with --reverse_input_channels argument specified. For more information about the argument, refer to When to Reverse Input Channels section of Converting a Model to Intermediate Representation (IR).

### Per-layer performance and logging
The application also collects per-layer Performance Measurement (PM) counters for each executed infer request if you enable statistics dumping by setting the `-report_type` parameter to one of the possible values:

* `no_counters` report includes configuration options specified, resulting FPS and latency.
* `average_counters` report extends the `no_counters` report and additionally includes average PM counters values for each layer from the network.
* `detailed_counters` report extends the `average_counters` report and additionally includes per-layer PM counters and latency for each executed infer request.

Depending on the type, the report is stored to benchmark_no_counters_report.csv, benchmark_average_counters_report.csv, or benchmark_detailed_counters_report.csv file located in the path specified in -report_folder. The application also saves executable graph information serialized to an XML file if you specify a path to it with the -exec_graph_path parameter.

### <a name="all-configuration-options"></a> All configuration options
Running the application with the `-h` or `--help` option yields the following usage message:

```
benchmark_app -h
[Step 1/11] Parsing and validating input arguments
usage: benchmark_app [-h [HELP]] [-i PATHS_TO_INPUT [PATHS_TO_INPUT ...]] -m PATH_TO_MODEL [-d TARGET_DEVICE] [-extensions PATH_TO_EXTENSIONS] [-c PATH_TO_CLDNN_CONFIG] [-hint {throughput,latency,none}]
                     [-api {sync,async}] [-niter NUMBER_ITERATIONS] [-nireq NUMBER_INFER_REQUESTS] [-b BATCH_SIZE] [-stream_output [STREAM_OUTPUT]] [-t TIME] [-progress [PROGRESS]] [-shape SHAPE]
                     [-data_shape DATA_SHAPE] [-layout LAYOUT] [-nstreams NUMBER_STREAMS]
                     [--latency_percentile {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100}]
                     [-nthreads NUMBER_THREADS] [-pin {YES,NO,NUMA,HYBRID_AWARE}] [-exec_graph_path EXEC_GRAPH_PATH] [-pc [PERF_COUNTS]] [-pcseq [PCSEQ]]
                     [-inference_only [INFERENCE_ONLY]] [-report_type {no_counters,average_counters,detailed_counters}] [-report_folder REPORT_FOLDER] [-dump_config DUMP_CONFIG]
                     [-load_config LOAD_CONFIG] [-infer_precision INFER_PRECISION] [-ip {u8,U8,f16,FP16,f32,FP32}] [-op {u8,U8,f16,FP16,f32,FP32}] [-iop INPUT_OUTPUT_PRECISION] [-cdir CACHE_DIR] [-lfile [LOAD_FROM_FILE]]
                     [-iscale INPUT_SCALE] [-imean INPUT_MEAN]

Options:
  -h [HELP], --help [HELP]
                        Show this help message and exit.
  -i PATHS_TO_INPUT [PATHS_TO_INPUT ...], --paths_to_input PATHS_TO_INPUT [PATHS_TO_INPUT ...]
                        Optional. Path to a folder with images and/or binaries or to specific image or binary file.It is also allowed to map files to network inputs:
                        input_1:file_1/dir1,file_2/dir2,input_4:file_4/dir4 input_2:file_3/dir3
  -m PATH_TO_MODEL, --path_to_model PATH_TO_MODEL
                        Required. Path to an .xml/.onnx file with a trained model or to a .blob file with a trained compiled model.
  -d TARGET_DEVICE, --target_device TARGET_DEVICE
                        Optional. Specify a target device to infer on (the list of available devices is shown below). Default value is CPU. Use '-d HETERO:<comma separated devices list>' format to
                        specify HETERO plugin. Use '-d MULTI:<comma separated devices list>' format to specify MULTI plugin. The application looks for a suitable plugin for the specified device.
  -extensions PATH_TO_EXTENSIONS, --extensions PATH_TO_EXTENSIONS
                        Optional. Path or a comma-separated list of paths to libraries (.so or .dll) with extensions.
  -c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
                        Optional. Required for GPU custom kernels. Absolute path to an .xml file with the kernels description.
  -hint {throughput,latency,none}, --perf_hint {throughput,latency,none}
                        Optional. Performance hint (latency or throughput or none). Performance hint allows the OpenVINO device to select the right network-specific settings. 'throughput': device
                        performance mode will be set to THROUGHPUT. 'latency': device performance mode will be set to LATENCY. 'none': no device performance mode will be set. Using explicit 'nstreams'
                        or other device-specific options, please set hint to 'none'
  -api {sync,async}, --api_type {sync,async}
                        Optional. Enable using sync/async API. Default value is async.
  -niter NUMBER_ITERATIONS, --number_iterations NUMBER_ITERATIONS
                        Optional. Number of iterations. If not specified, the number of iterations is calculated depending on a device.
  -nireq NUMBER_INFER_REQUESTS, --number_infer_requests NUMBER_INFER_REQUESTS
                        Optional. Number of infer requests. Default value is determined automatically for device.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Batch size value. If not specified, the batch size value is determined from Intermediate Representation
  -stream_output [STREAM_OUTPUT]
                        Optional. Print progress as a plain text. When specified, an interactive progress bar is replaced with a multi-line output.
  -t TIME, --time TIME  Optional. Time in seconds to execute topology.
  -progress [PROGRESS]  Optional. Show progress bar (can affect performance measurement). Default values is 'False'.
  -shape SHAPE          Optional. Set shape for input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size.This parameter affect model Parameter shape, can be
                        dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.
  -data_shape DATA_SHAPE
                        Optional. Optional if network shapes are all static (original ones or set by -shape).Required if at least one input shape is dynamic and input images are not provided.Set shape
                        for input tensors. For example, "input1[1,3,224,224][1,3,448,448],input2[1,4][1,8]" or "[1,3,224,224][1,3,448,448] in case of one input size.
  -layout LAYOUT        Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
  -nstreams NUMBER_STREAMS, --number_streams NUMBER_STREAMS
                        Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD (for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>). Default value is determined automatically for a device. Please note that although the automatic selection usually provides a reasonable performance, it still may be
                        non - optimal for some cases, especially for very small networks. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the
                        number of streams should be set to 1. See samples README for more details.
  --latency_percentile {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100}
                        Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).
  -nthreads NUMBER_THREADS, --number_threads NUMBER_THREADS
                        Number of threads to use for inference on the CPU, GNA (including HETERO and MULTI cases).
  -pin {YES,NO,NUMA,HYBRID_AWARE}, --infer_threads_pinning {YES,NO,NUMA,HYBRID_AWARE}
                        Optional. Enable threads->cores ('YES' which is OpenVINO runtime's default for conventional CPUs), threads->(NUMA)nodes ('NUMA'), threads->appropriate core types
                        ('HYBRID_AWARE', which is OpenVINO runtime's default for Hybrid CPUs) or completely disable ('NO') CPU threads pinning for CPU-involved inference.
  -exec_graph_path EXEC_GRAPH_PATH, --exec_graph_path EXEC_GRAPH_PATH
                        Optional. Path to a file where to store executable graph information serialized.
  -pc [PERF_COUNTS], --perf_counts [PERF_COUNTS]
                        Optional. Report performance counters.
  -pcseq [PCSEQ], --pcseq [PCSEQ]
                        Optional. Report latencies for each shape in -data_shape sequence.
  -inference_only [INFERENCE_ONLY], --inference_only [INFERENCE_ONLY]
                        Optional. If true inputs filling only once before measurements (default for static models), else inputs filling is included into loop measurement (default for dynamic models)
  -report_type {no_counters,average_counters,detailed_counters}, --report_type {no_counters,average_counters,detailed_counters}
                        Optional. Enable collecting statistics report. "no_counters" report contains configuration options specified, resulting FPS and latency. "average_counters" report extends
                        "no_counters" report and additionally includes average PM counters values for each layer from the network. "detailed_counters" report extends "average_counters" report and
                        additionally includes per-layer PM counters and latency for each executed infer request.
  -report_folder REPORT_FOLDER, --report_folder REPORT_FOLDER
                        Optional. Path to a folder where statistics report is stored.
  -dump_config DUMP_CONFIG
                        Optional. Path to JSON file to dump OpenVINO parameters, which were set by application.
  -load_config LOAD_CONFIG
                        Optional. Path to JSON file to load custom OpenVINO parameters. Please note, command line parameters have higher priority then parameters from configuration file.
  -infer_precision INFER_PRECISION
                        Optional. Hint to specifies inference precision. Example: -infer_precision CPU:bf16,GPU:f32'.
  -ip {u8,U8,f16,FP16,f32,FP32}, --input_precision {u8,U8,f16,FP16,f32,FP32}
                        Optional. Specifies precision for all input layers of the network.
  -op {u8,U8,f16,FP16,f32,FP32}, --output_precision {u8,U8,f16,FP16,f32,FP32}
                        Optional. Specifies precision for all output layers of the network.
  -iop INPUT_OUTPUT_PRECISION, --input_output_precision INPUT_OUTPUT_PRECISION
                        Optional. Specifies precision for input and output layers by name. Example: -iop "input:f16, output:f16". Notice that quotes are required. Overwrites precision from ip and op
                        options for specified layers.
  -cdir CACHE_DIR, --cache_dir CACHE_DIR
                        Optional. Enable model caching to specified directory
  -lfile [LOAD_FROM_FILE], --load_from_file [LOAD_FROM_FILE]
                        Optional. Loads model from file directly without read_network.
  -iscale INPUT_SCALE, --input_scale INPUT_SCALE
                        Optional. Scale values to be used for the input image per channel. Values to be provided in the [R, G, B] format. Can be defined for desired input of the model. Example: -iscale
                        data[255,255,255],info[255,255,255]
  -imean INPUT_MEAN, --input_mean INPUT_MEAN
                        Optional. Mean values to be used for the input image per channel. Values to be provided in the [R, G, B] format. Can be defined for desired input of the model. Example: -imean
                        data[255,255,255],info[255,255,255]
```

Running the application with the empty list of options yields the usage message given above and an error message.

### More information on inputs
The benchmark tool supports topologies with one or more inputs. If a topology is not data sensitive, you can skip the input parameter, and the inputs will be filled with random values. If a model has only image input(s), provide a folder with images or a path to an image as input. If a model has some specific input(s) (besides images), please prepare a binary file(s) that is filled with data of appropriate precision and provide a path to it as input. If a model has mixed input types, the input folder should contain all required files. Image inputs are filled with image files one by one. Binary inputs are filled with binary inputs one by one.

## Examples of Running the Tool
This section provides step-by-step instructions on how to run the Benchmark Tool with the `asl-recognition` Intel model on CPU or GPU devices. It uses random data as the input.

> **NOTE**: Internet access is required to execute the following steps successfully. If you have access to the Internet through a proxy server only, please make sure that it is configured in your OS environment.

1. Install OpenVINO Development Tools (if it hasn't been installed already):
   ```sh
   pip install openvino-dev
   ```

2. Download the model using omz_downloader, specifying the model name and directory to download the model to:
   ```sh
   omz_downloader --name asl-recognition-0004 --precisions FP16 --output_dir omz_models
   ```

3. Run the tool, specifying the location of the model .xml file, the device to perform inference on, and with a performance hint. The following commands demonstrate examples of how to run the Benchmark Tool in latency mode on CPU and throughput mode on GPU devices:

   * On CPU (latency mode):
   ```sh
   benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   ```

   * On GPU (throughput mode):
   ```sh
   benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d GPU -hint throughput
   ```

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.
Additionally, if you set the `-report_type` parameter, the application outputs a statistics report. If you set the `-pc` parameter, the application outputs performance counters. If you set `-exec_graph_path`, the application reports executable graph information serialized. All measurements including per-layer PM counters are reported in milliseconds.

An example of the information output when running benchmark_app on CPU in latency mode is shown below:

   ```sh
   benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   ```

   ```sh
   [Step 1/11] Parsing and validating input arguments
   [ INFO ] Parsing input parameters
   [ INFO ] Input command: /home/openvino/tools/benchmark_tool/benchmark_app.py -m omz_models/intel/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -hint latency
   [Step 2/11] Loading OpenVINO Runtime
   [ INFO ] OpenVINO:
   [ INFO ] Build ................................. 2022.3.0-7750-c1109a7317e-feature/py_cpp_align
   [ INFO ]
   [ INFO ] Device info:
   [ INFO ] CPU
   [ INFO ] Build ................................. 2022.3.0-7750-c1109a7317e-feature/py_cpp_align
   [ INFO ]
   [ INFO ]
   [Step 3/11] Setting device configuration
   [Step 4/11] Reading model files
   [ INFO ] Loading model files
   [ INFO ] Read model took 147.82 ms
   [ INFO ] Original model I/O parameters:
   [ INFO ] Model inputs:
   [ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / {1,3,16,224,224}
   [ INFO ] Model outputs:
   [ INFO ]     output (node: output) : f32 / [...] / {1,100}
   [Step 5/11] Resizing model to match image sizes and given batch
   [ INFO ] Model batch size: 1
   [Step 6/11] Configuring input of the model
   [ INFO ] Model inputs:
   [ INFO ]     input (node: input) : f32 / [N,C,D,H,W] / {1,3,16,224,224}
   [ INFO ] Model outputs:
   [ INFO ]     output (node: output) : f32 / [...] / {1,100}
   [Step 7/11] Loading the model to the device
   [ INFO ] Compile model took 974.64 ms
   [Step 8/11] Querying optimal runtime parameters
   [ INFO ] Model:
   [ INFO ]   NETWORK_NAME: torch-jit-export
   [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 2
   [ INFO ]   NUM_STREAMS: 2
   [ INFO ]   AFFINITY: Affinity.CORE
   [ INFO ]   INFERENCE_NUM_THREADS: 0
   [ INFO ]   PERF_COUNT: False
   [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
   [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
   [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
   [Step 9/11] Creating infer requests and preparing input tensors
   [ WARNING ] No input files were given for input 'input'!. This input will be filled with random values!
   [ INFO ] Fill input 'input' with random values
   [Step 10/11] Measuring performance (Start inference asynchronously, 2 inference requests, limits: 60000 ms duration)
   [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
   [ INFO ] First inference took 38.41 ms
   [Step 11/11] Dumping statistics report
   [ INFO ] Count:        5380 iterations
   [ INFO ] Duration:     60036.78 ms
   [ INFO ] Latency:
   [ INFO ]    Median:     22.04 ms
   [ INFO ]    Average:    22.09 ms
   [ INFO ]    Min:        20.78 ms
   [ INFO ]    Max:        33.51 ms
   [ INFO ] Throughput:   89.61 FPS
   ```

The Benchmark Tool can also be used with dynamically shaped networks to measure expected inference time for various input data shapes. See the -shape and -data_shape argument descriptions in the <a href="#all-configuration-options">All configuration options</a> section to learn more about using dynamic shapes. Here is a command example for using benchmark_app with dynamic networks and a portion of the resulting output:

   ```sh
   benchmark_app -m omz_models/intel/asl-recognition-0004/FP16/asl-recognition-0004.xml -d CPU -shape [-1,3,16,224,224] -data_shape [1,3,16,224,224][2,3,16,224,224][4,3,16,224,224] -pcseq
   ```

   ```sh
   [Step 9/11] Creating infer requests and preparing input tensors
  [ WARNING ] No input files were given for input 'input'!. This input will be filled with random values!
  [ INFO ] Fill input 'input' with random values
  [ INFO ] Defined 3 tensor groups:
  [ INFO ]         input: {1, 3, 16, 224, 224}
  [ INFO ]         input: {2, 3, 16, 224, 224}
  [ INFO ]         input: {4, 3, 16, 224, 224}
   [Step 10/11] Measuring performance (Start inference asynchronously, 11 inference requests, limits: 60000 ms duration)
   [ INFO ] Benchmarking in full mode (inputs filling are included in measurement loop).
   [ INFO ] First inference took 201.15 ms
   [Step 11/11] Dumping statistics report
   [ INFO ] Count:        2811 iterations
   [ INFO ] Duration:     60271.71 ms
   [ INFO ] Latency:
   [ INFO ]    Median:     207.70 ms
   [ INFO ]    Average:    234.56 ms
   [ INFO ]    Min:        85.73 ms
   [ INFO ]    Max:        773.55 ms
   [ INFO ] Latency for each data shape group:
   [ INFO ] 1. input: {1, 3, 16, 224, 224}
   [ INFO ]    Median:     118.08 ms
   [ INFO ]    Average:    115.05 ms
   [ INFO ]    Min:        85.73 ms
   [ INFO ]    Max:        339.25 ms
   [ INFO ] 2. input: {2, 3, 16, 224, 224}
   [ INFO ]    Median:     207.25 ms
   [ INFO ]    Average:    205.16 ms
   [ INFO ]    Min:        166.98 ms
   [ INFO ]    Max:        545.55 ms
   [ INFO ] 3. input: {4, 3, 16, 224, 224}
   [ INFO ]    Median:     384.16 ms
   [ INFO ]    Average:    383.48 ms
   [ INFO ]    Min:        305.51 ms
   [ INFO ]    Max:        773.55 ms
   [ INFO ] Throughput:   108.82 FPS
   ```

## See Also
* [Using OpenVINO Samples](../../docs/OV_Runtime_UG/Samples_Overview.md)
* [Model Optimizer](../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
* [Model Downloader](@ref omz_tools_downloader)
