# RAG-Powered-AI-Assistant

Welcome to the RAG-Powered AI Assistant repository! This project demonstrates how to build a Retrieval-Augmented Generation (RAG) powered AI assistant.

## Preparation

1. Clone the repository:

   ```bash
   git clone https://github.com/HiIAmTzeKean/RAG-Powered-AI-Assistant
   ```

2. You will have to download the knowledge base of publications from [Ready Tensor](https://drive.google.com/drive/folders/1HAqLXL2W-sh8hqoBb1iSauJ_0wZVRxB9?usp=sharing) and place it under `/data` directory first.

3. Install the required dependencies using `uv`:

    ```bash
   uv sync
   ```

## Required environment variables

- `MISTRAL_API_KEY`: Your Mistral API key.
- `HF_KEY`: Your Hugging Face API key

## Usage

To run the AI assistant, use the following command after ensuring that the environment variables are set:

```bash
uv run main.py "<your question here>"
```
