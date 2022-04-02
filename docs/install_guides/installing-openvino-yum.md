# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using YUM Repository {#openvino_docs_install_guides_installing_openvino_yum}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux distributed through the YUM repository.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please review the content inside the `<INSTALL_DIR>/licensing` folder for more details.

## System Requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html).

**Operating systems**

- Red Hat Enterprise Linux 8, 64-bit

**Software**

- [CMake 3.13 or higher, 64-bit](https://cmake.org/download/)
- GCC 8.2.0
- [Python 3.6 - 3.9, 64-bit](https://www.python.org/downloads/windows/)

## Install OpenVINO Runtime

### Step 1: Set Up the Repository

1. Create the YUM repo file in the `/tmp` directory as a normal user:
   ```
   tee > /tmp/openvino-2022.repo << EOF
   [OpenVINO]
   name=Intel(R) Distribution of OpenVINO 2022
   baseurl=https://yum.repos.intel.com/openvino/2022
   enabled=1
   gpgcheck=1
   repo_gpgcheck=1
   gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
   EOF
   ```
2.	Move the new openvino-2022.repo file to the YUM configuration directory `/etc/yum.repos.d`:
   ```sh
   sudo mv /tmp/openvino-2022.repo /etc/yum.repos.d
   ```
3.	Verify that the new repo is properly setup by running the following command:
   ```sh
   yum repolist | grep -i openvino
   ```
    You will see the available list of packages.


To list available OpenVINO packages, use the following command:
```
yum list 'openvino*'
```

### Step 2: Install OpenVINO Runtime Using the YUM Package Manager

Intel® Distribution of OpenVINO™ toolkit will be installed in: `/opt/intel/openvino_<VERSION>.<UPDATE>.<PATCH>`

A symlink will be created: `/opt/intel/openvino_<VERSION>`

You can select one of the following procedures according to your need:

#### To Install the Latest Version

Run the following command:
```sh
sudo yum install openvino
```

#### To Install a Specific Version

Run the following command:
```sh
sudo yum install openvino-<VERSION>.<UPDATE>.<PATCH>
```

For example:
```sh
sudo yum install openvino-2022.1.0
```

#### To Check for Installed Packages and Version

Run the following command:
```sh
yum list installed 'openvino*'
```

#### To Uninstall the Latest Version

Run the following command:
```sh
sudo yum autoremove openvino
```

#### To Uninstall a Specific Version

Run the following command:
```sh
sudo yum autoremove openvino-<VERSION>.<UPDATE>.<PATCH>
```

### Step 3 (Optional): Install OpenCV from YUM

OpenCV is necessary to run C++ demos from Open Model Zoo. Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. OpenVINO provides a package to install OpenCV from YUM:

#### To Install the Latest Version of OpenCV

Run the following command:
```sh
sudo yum install openvino-opencv
```

#### To Install a Specific Version of OpenCV

Run the following command:
```sh
sudo yum install openvino-opencv-<VERSION>.<UPDATE>.<PATCH>
```

### Step 4 (Optional): Install Software Dependencies

After you have installed OpenVINO Runtime, if you decided to [install OpenVINO Model Development Tools](installing-model-dev-tools.md), make sure that you install external software dependencies first. 

Refer to <a href="openvino_docs_install_guides_installing_openvino_linux.html#install-external-dependencies">Install External Software Dependencies</a> for detailed steps.

### Step 5 (Optional): Configure Inference on Non-CPU Devices

@sphinxdirective

.. tab:: GNA

   To enable the toolkit components to use Intel® Gaussian & Neural Accelerator (GNA) on your system, follow the steps in :ref:`GNA Setup Guide <gna guide>`.
   
.. tab:: GPU

   To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide>`.

.. tab:: NCS 2

   To perform inference on Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X VPU, follow the steps on :ref:`NCS2 Setup Guide <ncs guide>`.
   <!--For more details, see the `Get Started page for Intel® Neural Compute Stick 2 <https://software.intel.com/en-us/neural-compute-stick/get-started>`.-->

.. tab:: VPU

   To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the :ref:`VPU Configuration Guide <vpu guide>`.
   After configuration is done, you are ready to run the verification scripts with the HDDL Plugin for your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs. 

   .. warning::
      While working with either HDDL or NCS, choose one of them as they cannot run simultaneously on the same machine.

@endsphinxdirective


## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/latest/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
