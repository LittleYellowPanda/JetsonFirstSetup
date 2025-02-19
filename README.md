# NVIDIA Jetson Orin AGX (the proper setup)

> I wrote this page because setting up the Jetson Orin AGX is a pain in the a**. Mostly to have the proper packages for CUDA, and using GPU with Deep Learning Frameworks such as PyTorch and TensorFlow.

## Flash the proper way

- **For brand new devices:**  
  Use the SDK Manager to flash your device.
  
- **For already flashed devices:**  
  Put it in “force recovery mode”:
  - Run the command:  
    ```bash
    sudo reboot --force forced-recovery
    ```
  - Or press the middle button and the force recovery button simultaneously ([see the NVIDIA doc](https://developer.nvidia.com)).

Next, use a laptop (or any computer) running a **Linux OS** (this setup won’t work on other OSes). Install the **SDK Manager**—NVIDIA’s GUI tool to install and flash their devices—and make sure to set up an **NVIDIA Developer Account**!

Now, connect your device with a **USB A/C cable** (with the USB-C end plugged into the Orin). Verify the connection by running:
  
```bash
lsusb | grep Nvidia
```

Flash your device following the on-screen instructions, and that’s it!  
_Obviously, that was the easy part._

---

## How I set up my developing environment

> **Tip:** If you can afford it, put your Orin in **MAXN power mode** (everything will be smoother).

Once you're on the Desktop, you'll notice that you have very few applications installed.  
To automate the repetitive setup tasks, I created the following **bash script**:

```bash
#!/bin/bash

# Update package lists
sudo apt update -y
sudo apt upgrade -y

# Install Firefox and Chromium
sudo apt install -y firefox chromium-browser

# Install nano & other useful packages
sudo apt install -y nano
sudo apt install tree

# Install UV python package manager
curl -sSL https://astral.sh/uv/install.sh | bash 
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc 

# Install Docker
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to the docker group to allow running without sudo
sudo usermod -aG docker $USER

# Install Visual Studio Code
sudo apt install -y wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/packages.microsoft.gpg > /dev/null
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
sudo apt update -y
sudo apt install -y code

# Give VS Code permissions to access Docker
sudo setfacl -m user:$USER:rw /var/run/docker.sock

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Notify the user to restart the session to apply Docker group changes
echo "Please log out and log back in for Docker group changes to take effect."

echo "Setup complete!"
```

### Verify Installations

- **Check if Docker is installed and properly configured** by running:
  
  ```bash
  docker run hello-world
  ```
  
  If it works correctly, it should pull the corresponding image from Docker Hub.

- **For VS Code**, simply type `code` in your terminal or launch it from your application menu.

---

## The tough step… Searching for CUDA

# The Tough Step… Searching for CUDA

The main reason you spent (or rather convinced your company to spend) **3000€** on this little box is for its **GPU** (and the Jetson AI environment that comes with it). It would be a real shame if you couldn't access it, right?  

**AND** the only way to make your GPU play nice with all your Python or C++ software… is to use **CUDA**.  

> 💡 **CUDA** is a critical API that allows deep learning frameworks to access and leverage NVIDIA GPUs, massively accelerating machine learning tasks.

### Let's Check if CUDA is Installed

First, check if CUDA is properly installed on your machine. Run:

```bash
ls /usr/local/cuda
```

If CUDA is installed, this command should display its contents. If the directory doesn’t exist, CUDA might not be installed correctly.  

### **If CUDA is Missing:**
```bash
sudo apt update
sudo apt install nvidia-jetpack

# Just in case...
sudo reboot
```

And if you **still** can’t find it… well, that’s normal. You just don’t know where to look yet (don’t worry, I’m also blind sometimes).  

---

## **How to Search for CUDA**

### 1️⃣ Check the Default CUDA Installation Path  

```bash
ls /usr/local | grep cuda
```
If CUDA is installed, you should see a folder like `cuda-12.2` (or another version number).  

### 2️⃣ Verify if CUDA Binaries Exist  

```bash
which nvcc
```

### 3️⃣ Check if CUDA is in Your `PATH`  

```bash
echo $PATH | grep cuda
```

If none of these commands give useful results, it's likely that CUDA isn’t properly set in your environment variables.  

---

## **Add CUDA to Your Path**

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Now, check if CUDA is properly installed:  

```bash
nvcc --version
```

If everything is working, you should see something like this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

---

## **Final Test: Running CUDA Samples**

```bash
# Check if CUDA samples can compile and run:
cuda-install-samples-$(nvcc --version | grep -oP '\d+\.\d+') ~/
cd ~/NVIDIA_CUDA-*/Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

If everything is set up correctly, this should detect your GPU:

```
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Orin"
  CUDA Driver Version / Runtime Version          12.6 / 12.6
  CUDA Capability Major/Minor version number:    8.7
  Total amount of global memory:                 62841 MBytes (65893089280 bytes)
  (016) Multiprocessors, (128) CUDA Cores/MP:    2048 CUDA Cores
  GPU Max Clock rate:                            1300 MHz (1.30 GHz)
  Memory Clock rate:                             1300 MHz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  ...
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.6, CUDA Runtime Version = 12.6, NumDevs = 1
Result = PASS
```

---

## **Troubleshooting with `deviceQuery`**

If you get **"directory not found"**, don’t panic—I got you.  

Go and locate your CUDA samples at `/usr/local/cuda/samples/`.  

If it's missing, clone the official NVIDIA CUDA samples repository:  

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
```

Navigate to the `deviceQuery` sample directory:

```bash
cd cuda-samples/Samples/1_Utilities/deviceQuery
```

> 💡 **Why bother with `deviceQuery` if `nvcc --version` works fine?**  
> Because `nvcc --version` only tells you if the compiler is installed, **not** if your GPU is actually working properly!

If `deviceQuery` is missing, install **CMake**:

```bash
sudo apt-get install cmake
```

Then, create a build directory and compile `deviceQuery`:

```bash
mkdir build
cd build

# Generate Makefiles with CMake
cmake ..

# Compile and run
make
./deviceQuery
```

🎉 **Tadaaaaaaa!** Now your GPU is ready to roll.

---

# **How to Set Up PyTorch and TensorFlow to Use CUDA**

### **Prerequisites: JetPack Must Be Installed!**

First, check if JetPack is installed (if you followed this guide, it should be, but hey—double-checking never hurts):

```bash
dpkg -l | grep 'nvidia-l4t-core'
sudo apt-cache show nvidia-jetpack

# Or a quick check:
jetson-release
```

If JetPack is **not** installed… go back to [this step](https://www.notion.so/NVIDIA-Jetson-Orin-AGX-the-proper-setup-19fb07e3dae080fa8542dbdb9dd24477?pvs=21) (and I’m truly sorry for your pain).  

If JetPack **is** installed, continue! 🚀  

---

## **Create a Virtual Environment (Recommended)**  

A good practice is to always develop in a **virtual environment**. It keeps each project isolated, preventing dependency conflicts.  

I recommend **uv**, a new package manager built in Rust that is **blazing fast**. It's similar to Poetry but much lighter and more efficient.

```bash
uv init FER
```

This initializes a virtual environment in your `FER` project folder.

---

## **Install PyTorch with CUDA Support**  

```bash
sudo apt-get update -y
sudo apt-get install -y python3-pip libopenblas-dev
```

> If you're installing **PyTorch 24.06 or later**, you **must** install `cusparselt` first:  

```bash
wget https://raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
export CUDA_VERSION=12.1  # Example CUDA version
bash ./install_cusparselt.sh
```

Now, check your JetPack & CUDA versions and find a compatible PyTorch `.whl` file:  

- [NVIDIA PyTorch Forum](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)  
- [Jetson-AI PyTorch Packages](https://pypi.jetson-ai-lab.dev/jp6/cu126)  

For example, if you have:  

- **JetPack 6.1**
- **CUDA 12.6**

Then download the correct PyTorch wheel file and install it:  

```bash
export TORCH_INSTALL=/home/nvidia/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl

uv pip install --upgrade pip
uv pip install numpy==1.26.1
uv pip install --no-cache-dir $TORCH_INSTALL
```

---

## **Verify PyTorch Installation**  

Run this Python command:

```python
import torch
print('Torch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('GPU Count:', torch.cuda.device_count())
print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')
```

If you see CUDA enabled and your GPU name listed—congrats, you're ready to train some models! 🎉  

---

## **References**
- [NVIDIA Jetson Orin AGX Setup Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)
- [Official PyTorch for Jetson Guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
