# Converting a Kaldi* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi}

A summary of the steps for optimizing and deploying a model that was trained with Kaldi\*:

1. [Configure the Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md) for Kaldi\*.
2. [Convert a Kaldi\* Model](#Convert_From_Kaldi) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
3. Test the model in the Intermediate Representation format using the [OpenVINO™ Runtime](../../../OV_Runtime_UG/openvino_intro.md) in the target environment via provided [OpenVINO Samples](../../../OV_Runtime_UG/Samples_Overview.md).
4. [Integrate](../../../OV_Runtime_UG/Samples_Overview.md) the [OpenVINO™ Runtime](../../../OV_Runtime_UG/openvino_intro.md) in your application to deploy the model in the target environment.

> **NOTE**: The Model Optimizer supports the [nnet1](http://kaldi-asr.org/doc/dnn1.html) and [nnet2](http://kaldi-asr.org/doc/dnn2.html) formats of Kaldi models. Support of the [nnet3](http://kaldi-asr.org/doc/dnn3.html) format is limited.

## Convert a Kaldi* Model <a name="Convert_From_Kaldi"></a>

To convert a Kaldi\* model, run Model Optimizer with the path to the input model `.nnet` or `.mdl` file and to an output directory where you have write permissions:
```sh
mo --input_model <INPUT_MODEL>.nnet --output_dir <OUTPUT_MODEL_DIR>
```

Two groups of parameters are available to convert your model:

* Framework-agnostic parameters are used to convert a model trained with any supported framework. For details, see the General Conversion Parameters section on the [Converting a Model to Intermediate Representation (IR)](Converting_Model.md) page.
* [Kaldi-specific parameters](#kaldi_specific_conversion_params) are used to convert only Kaldi\* models.

### Using Kaldi\*-Specific Conversion Parameters <a name="kaldi_specific_conversion_params"></a>

The following list provides the Kaldi\*-specific parameters.

```sh
Kaldi-specific parameters:
  --counts COUNTS       A file name with full path to the counts file or empty string to utilize count values from the model file
  --remove_output_softmax
                        Removes the Softmax that is the output layer
  --remove_memory       Remove the Memory layer and add new inputs and outputs instead
```

### Examples of CLI Commands

* To launch the Model Optimizer for the wsj_dnn5b_smbr model with the specified `.nnet` file and an output directory where you have write permissions:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet --output_dir <OUTPUT_MODEL_DIR>
   ```

* To launch the Model Optimizer for the wsj_dnn5b_smbr model with existing file that contains counts for the last layer with biases and a writable output directory:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts --output_dir <OUTPUT_MODEL_DIR>
   ```

  * The Model Optimizer normalizes сounts in the following way:
	\f[
	S = \frac{1}{\sum_{j = 0}^{|C|}C_{j}}
	\f]
	\f[
	C_{i}=log(S*C_{i})
	\f]
	where \f$C\f$ - the counts array, \f$C_{i} - i^{th}\f$ element of the counts array,
	\f$|C|\f$ - number of elements in the counts array;
  * The normalized counts are subtracted from biases of the last or next to last layer (if last layer is SoftMax).

  > **NOTE:** Model Optimizer will show warning if model contains counts values inside model and `--counts` option is not used.

* If you want to remove the last SoftMax layer in the topology, launch the Model Optimizer with the
`--remove_output_softmax` flag:
   ```sh
   mo --input_model wsj_dnn5b_smbr.nnet --counts wsj_dnn5b_smbr.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>_
   ```

The Model Optimizer finds the last layer of the topology and removes this layer only if it is a SoftMax layer.

  > **NOTE**: Model Optimizer can remove SoftMax layer only if the topology has one output.

  > **NOTE**: For sample inference of Kaldi models, you can use the OpenVINO Speech Recognition sample application. The sample supports models with one output. If your model has several outputs, specify the desired one with the `--output` option.

 If you want to convert a model for inference on Intel® Movidius™ Myriad™, use the `--remove_memory` option.
It removes Memory layers from the IR. Instead of it, additional inputs and outputs appear in the IR.
The Model Optimizer outputs the mapping between inputs and outputs. For example:
```sh
[ WARNING ]  Add input/output mapped Parameter_0_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out -> Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out
[ WARNING ]  Add input/output mapped Parameter_1_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out -> Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out
[ WARNING ]  Add input/output mapped Parameter_0_for_iteration_Offset_fastlstm3.c_trunc__3390 -> Result_for_iteration_Offset_fastlstm3.c_trunc__3390
```
 Based on this mapping, link inputs and outputs in your application manually as follows:

1. Initialize inputs from the mapping as zeros in the first frame of an utterance.
2. Copy output blobs from the mapping to the corresponding inputs. For example, data from `Result_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out`
must be copied to `Parameter_0_for_Offset_fastlstm2.r_trunc__2Offset_fastlstm2.r_trunc__2_out`.

## Supported Kaldi\* Layers
Refer to [Supported Framework Layers ](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
