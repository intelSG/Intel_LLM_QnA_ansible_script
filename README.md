# Intel_LLM_QnA_ansible_script (RAG Q&A Application with Ansible and Docker)

This project combines Ansible and Docker to automate the setup of an environment for running an Intel RAG (Retrieval Augmented Generation) application. The Ansible script orchestrates the installation of dependencies, while the Docker image encapsulates the application and its environment.

# Table of Contents
1. [Ansible Setup for Intel RAG Environment](#ansible-setup-for-intel-rag-environment)
    - [Prerequisites](#prerequisites)
    - [Usage](#usage)
    - [Playbook Details](#playbook-details)
        * [Variables](#variables)
        * [Tasks](#tasks)
    - [Note](#note)
2. [Allen_AI Q&A Application Docker Image](#allen_ai-qa-application-docker-image)
    - [Usage](#usage-1)
        * [Build the Docker Image](#build-the-docker-image)
        * [Run the Docker Container](#run-the-docker-container)
    - [Container Details](#container-details)
        * [Base Image](#base-image)
        * [Working Directory](#working-directory)
        * [Conda Environment](#conda-environment)
        * [Dependencies](#dependencies)
        * [Exposed Port](#exposed-port)
        * [Application Execution](#application-execution)
        * [Running the Application](#running-the-application)


<a name="ansible-setup-for-intel-rag-environment"></a>
## Ansible Setup for Intel RAG Environment

This Ansible script automates the setup of an environment for running an Intel RAG (Retrieval Augmented Generation) application. The environment includes dependencies such as Miniconda, Conda environments, Git, and specific Python libraries for PyTorch and Intel Extension for PyTorch.

### Prerequisites

- Ansible installed on the control machine.
- Target machine(s) accessible and configured for SSH access.

### Usage

1. Clone the Ansible repository to your local machine:

```bash
git clone https://github.com/Yuandjom/Intel_LLM_QnA_ansible_script.git
```
2. Navigate to the cloned directory:

```bash
cd Intel_LLM_QnA_ansible_script
```

3. Update the `hosts.ini` file with the IP address and hostname of your target machine(s).
4. Adjust variables in the Ansible playbook (setup_RAG.yml) if needed, especially the `workspace_path` variable.
5. Run the Ansible playbook to set up the environment:

```bash
ansible-playbook -i hosts.ini setup_RAG.yml -vv
```
### Playbook Details
#### Variables
- `workspace_path`: Path to the main workspace on the target machine.
- `workspace_path_hf`: Path to a specific directory within the workspace.
- `conda_installer_url`: URL for the Miniconda installer script.
- `conda_env_file`: Local path to the Conda environment YAML file.
- `remote_conda_env_file:` Remote path for copying the Conda environment file.
- `conda_path`: Path to the Conda executable on the target machine.
- `local_directory`: Local path to the directory to be copied to the target machine.
- `local_directory_huggingFace`: Local path to another directory to be copied to the target machine.

#### Tasks
1. Install Git: Ensures that Git is installed on the target machine.

2. Create Workspace Directory: Creates the main workspace directory.

3. Copy Directories to Remote Host: Copies specified local directories to the target machine.

4. Install Conda: Downloads and installs Miniconda on the target machine.

5. Set Conda Permissions: Adjusts permissions for Conda installation.

6. Copy Conda Environment File: Copies the Conda environment YAML file to the target machine.

7. Create Conda Environment: Creates a Conda environment based on the provided YAML file.

8. Install Additional Dependencies: Installs Git and additional dependencies required for the environment.

9. Find Python Executable in Conda Environment: Finds the Python executable in the Conda environment.

10. Install Pip in Conda Environment: Installs Pip in the Conda environment.

11. Get CPU Architecture Information: Retrieves information about the CPU architecture.

12. Check PyTorch and Intel Extension for PyTorch Versions: Checks and prints PyTorch and Intel Extension for PyTorch versions.

### Note
- The playbook is designed for Linux-based systems.
- Ensure that the SSH connection to the target machine is properly configured.
- Some tasks may require manual intervention or adjustments based on the target environment.

## Allen_AI Q&A Application Docker Image
This Docker image sets up an environment for running an Allen_AI-based Question & Answer (Q&A) application using the FARM and Haystack libraries. The image is built on top of the Miniconda3 base image and includes the necessary dependencies for the application.

### Usage

#### Build the Docker Image
To build the Docker image, use the following command:
```bash
sudo docker build -t allen_ai_app .
```

#### Run the Docker Container
To run the Docker container, execute the following command:
```bash
sudo docker run -it -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 allen_ai_app
```

Copy link into the command line 
```bash
https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip
```

This command maps port 7860 on the host to port 7860 on the container and sets the Gradio server to be accessible from all network interfaces.

#### Container Details

##### Base Image
The base image for this Docker container is continuumio/miniconda3, which provides a minimal Conda installation.

##### Working Directory
The working directory within the container is set to ```/home/test/QnA_RAG```.

##### Conda Environment
A Conda environment named `Allen_AI` is created with Python version 3.8. The environment is activated during the Docker build process and maintained for running the application.

##### Dependencies
The following dependencies are installed within the `Allen_AI` Conda environment:
- PyTorch, torchvision, and torchaudio
- FARM and Haystack libraries
- Gradio for creating interactive user interfaces

#### Exposed Port
Port 7860 is exposed to allow external access to the Gradio server, assuming that Gradio runs on this port within the application.

#### Application Execution
The container is configured to execute the Python script `Generative_QA_Haystack_PromptNode_CSV_database_llama2.py` within the specific directory `/home/test/QnA_RAG/QnA_fastRAG_CSV_LLAMA2` when the container starts.

#### Running the Application
After building and running the Docker container, you can access the Allen_AI Q&A application by navigating to `http://localhost:7860` in your web browser.

Feel free to customize the Dockerfile and script to suit your specific use case and application requirements.

**Note:** Certain files, including `tokenizer.json`, `pytorch_model-00001-of-00002.bin`, `pytorch_model-00002-of-00002.bin`, `model-00001-of-00002.safetensors`, and `model-00002-of-00002.safetensors` within the `Llama-2-7b-chat-hf` directory, were not pushed to the repository due to their large size. You may need to obtain these files separately if they are essential for your use case.



