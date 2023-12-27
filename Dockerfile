# Use a base image that includes conda https://hub.docker.com/r/continuumio/miniconda3/
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /home/test/QnA_RAG

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Create a conda environment based on the dependencies you've listed
RUN conda create -n Allen_AI python=3.8 --yes
RUN echo "source activate Allen_AI" > ~/.bashrc
ENV PATH /opt/conda/envs/Allen_AI/bin:$PATH

# Install the necessary packages
# RUN conda install -n Allen_AI -c conda-forge -c defaults --yes \
#     pip \
#     pytorch torchvision torch torchaudio \
#     && conda clean -afy
RUN conda install -n Allen_AI -c pytorch --yes \
        pytorch torchvision torchaudio \
    && conda clean -afy

# RUN conda install -n Allen_AI -c huggingface transformers --yes 

# Activate the conda environment
SHELL ["conda", "run", "-n", "Allen_AI", "/bin/bash", "-c"]

# Install the required pip packages
RUN pip install farm-haystack==1.17.2 gradio==3.40.1 faiss-cpu 'farm-haystack[sql]' 'farm-haystack[inference]'

# Make port 7860 available to the world outside this container
# Assuming Gradio runs on port 7860, change if different
EXPOSE 7860

# Change to the specific directory
WORKDIR /home/test/QnA_RAG/QnA_fastRAG_CSV_LLAMA2

# Run the python script
CMD ["python", "Generative_QA_Haystack_PromptNode_CSV_database_llama2.py"]
