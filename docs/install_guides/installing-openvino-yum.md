# Install Intel® Distribution of OpenVINO™ Toolkit for Linux Using YUM Repository {#openvino_docs_install_guides_installing_openvino_yum}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit for Linux distributed through the YUM repository.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. If you want to develop or optimize your models with OpenVINO, see [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

> **IMPORTANT**: By downloading and using this container and the included software, you agree to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf). Please review the content inside the `<INSTALL_DIR>/licensing` folder for more details.

## System Requirements

The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html).

**Operating systems**

- Red Hat Enterprise Linux 8, 64-bit

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

OpenCV is necessary to run C++ demos from Open Model Zoo. Some C++ samples and demos also use OpenCV as a dependency. OpenVINO provides a package to install OpenCV from YUM:

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

Refer to <a href="#install-external-dependencies">Install External Software Dependencies</a> for detailed steps.

## Configurations for Non-CPU Devices

If you are using Intel® Processor Graphics, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs or Intel® Neural Compute Stick 2, please follow the configuration steps in [Configurations for GPU](configurations-for-intel-gpu.md), [Configurations for VPU](installing-openvino-config-ivad-vpu.md) or [Configurations for NCS2](configurations-for-ncs2.md) accordingly.


## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>.
- OpenVINO™ toolkit online documentation: <https://docs.openvino.ai/>.
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [OpenVINO Runtime User Guide](../OV_Runtime_UG/OpenVINO_Runtime_User_Guide).
- For more information on Sample Applications, see the [OpenVINO Samples Overview](../OV_Runtime_UG/Samples_Overview.md).
- For IoT Libraries & Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
