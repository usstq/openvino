# Install Intel® Distribution of OpenVINO™ toolkit from Anaconda Cloud {#openvino_docs_install_guides_installing_openvino_conda}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux distributed through the Anaconda Cloud.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

## System Requirements

**Software**

 - [Anaconda distribution](https://www.anaconda.com/products/individual/)

**Operating Systems**

| Supported Operating System                                   | [Python Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8, 3.9                                  |
|   Ubuntu 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8, 3.9                                  |
|   Red Hat Enterprise Linux 8, 64-bit                         | 3.6, 3.7, 3.8, 3.9                                  |
|   macOS 10.15                                             | 3.6, 3.7, 3.8, 3.9                                  |
|   Windows 10, 64-bit                                        | 3.6, 3.7, 3.8, 3.9                                  |

## Install OpenVINO Runtime Using the Anaconda Package Manager

1. Set up the Anaconda environment (taking Python 3.7 for example): 
   ```sh
   conda create --name py37 python=3.7
   conda activate py37
   ```
2. Update Anaconda environment to the latest version:
   ```sh
   conda update --all
   ```
3. Install the Intel® Distribution of OpenVINO™ toolkit:
 - Ubuntu* 20.04 
   ```sh
   conda install openvino-ie4py-ubuntu20 -c intel
   ```
 - Ubuntu* 18.04
   ```sh
   conda install openvino-ie4py-ubuntu18 -c intel
   ```
 - Red Hat Enterprise Linux 8, 64-bit 
   ```sh
   conda install openvino-ie4py-rhel8 -c intel
   ```
 - Windows 10 and macOS
   ```sh
   conda install openvino-ie4py -c intel
   ```
4. Verify the package is installed:
   ```sh
   python -c "from openvino.runtime import Core"
   ```
   If installation was successful, you will not see any error messages (no console output).

Now you can start developing your application.


## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- OpenVINO™ toolkit online documentation: <https://docs.openvinotoolkit.ai>.
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [OpenVINO Runtime User Guide](../OV_Runtime_UG/OpenVINO_Runtime_User_Guide).
- For more information on Sample Applications, see the [OpenVINO Samples Overview](../OV_Runtime_UG/Samples_Overview.md).
- Intel® Distribution of OpenVINO™ toolkit Anaconda home page: [https://anaconda.org/intel/openvino-ie4py](https://anaconda.org/intel/openvino-ie4py)
