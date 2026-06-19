# Training Mamba on Google Colab with VS Code Remote SSH

This guide provides step-by-step instructions for training the Mamba model on a Google Colab GPU instance while using VS Code as your local development environment. This setup allows you to leverage Colab's free GPU resources for intensive training tasks and enjoy the full-featured editing and debugging experience of VS Code.

## 1. Prerequisites

- A Google account.
- [Visual Studio Code](https://code.visualstudio.com/) installed on your local machine.
- The [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension for VS Code.

## 2. Setting up the Colab Environment

1.  **Create a New Colab Notebook:**
    - Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
    - In the notebook, go to **Runtime > Change runtime type** and select **T4 GPU** as the hardware accelerator.

2.  **Mount Google Drive:**
    - Add the following code to a cell in your notebook and run it to mount your Google Drive. This will allow you to save your project files and model checkpoints persistently.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3.  **Install and Configure `colab-ssh`:**
    - `colab-ssh` is a Python package that simplifies the process of connecting to a Colab instance via SSH. Run the following code in a notebook cell to install it and launch the SSH service.

    ```python
    # Install colab-ssh
    !pip install colab-ssh

    # Launch colab-ssh
    from colab_ssh import launch_ssh_cloudflared
    launch_ssh_cloudflared(password="your_secure_password")
    ```

    - Replace `"your_secure_password"` with a strong password of your choice.
    - After running this cell, you will see the SSH connection details, including a hostname and port number. Keep this information handy.

## 3. Connecting VS Code to Colab

1.  **Configure SSH in VS Code:**
    - Open the Command Palette in VS Code (**Ctrl+Shift+P** or **Cmd+Shift+P**).
    - Type **"Remote-SSH: Open SSH Configuration File..."** and select the default `config` file.
    - Add a new SSH host entry using the details from the previous step:

    ```
    Host colab
        HostName <your_hostname>
        User root
        Port <your_port>
    ```

    - Replace `<your_hostname>` and `<your_port>` with the values provided by `colab-ssh`.

2.  **Connect to the Colab Instance:**
    - Open the Command Palette again and type **"Remote-SSH: Connect to Host..."**.
    - Select the **`colab`** host you just configured.
    - A new VS Code window will open, connected to your Colab instance. You will be prompted to enter the password you set earlier.

## 4. Setting up the Project in the Colab Environment

1.  **Clone the Repository:**
    - Open a new terminal in VS Code (**Terminal > New Terminal**).
    - Navigate to your mounted Google Drive and clone the project repository:

    ```bash
    cd /content/drive/MyDrive
    git clone https://github.com/Kaushikj-7/mamba.git
    cd mamba
    ```

2.  **Set up the Python Environment:**
    - Create and activate a Python virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    - Install the required Python packages using the `requirements.txt` file:

    ```bash
    export MAX_JOBS=1
    pip install -r requirements.txt
    ```

## 5. Running the Training

1.  **Start the Training:**
    - Execute the training script with the `mamba_125m.yaml` configuration:

    ```bash
    python src/training/train_mamba.py --config configs/mamba_125m.yaml
    ```

2.  **Resuming Training:**
    - The training script is configured to save checkpoints in the `checkpoints` directory. If your Colab session is interrupted, you can resume training from the last saved checkpoint by running the same command again. The script will automatically detect and load the latest checkpoint.

## 6. Monitoring and Evaluation

- You can monitor the training progress in the VS Code terminal.
- Once the training is complete, you can run the evaluation scripts as described in the `REPLICATION_GUIDE.md` to assess the model's performance.

## 7. Alternative Method: Running Directly in the Colab Notebook

If you prefer not to use SSH and VS Code, you can run the entire training process directly within the Google Colab notebook. This approach is simpler but offers a less integrated development experience.

### 7.1. Setup the Notebook

1.  **Create a New Colab Notebook:**
    - Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
    - Go to **Runtime > Change runtime type** and select **T4 GPU**.

2.  **Clone the Repository and Install Dependencies:**
    - Add the following code to a cell and run it:

    ```python
    # Clone the repository
    !git clone https://github.com/Kaushikj-7/mamba.git
    %cd mamba

    # Install dependencies
    !export MAX_JOBS=1
    !pip install -r requirements.txt
    ```

### 7.2. Prepare Data and Run Training

1.  **Run Scripts from Notebook Cells:**
    - You can run shell commands and Python scripts directly from notebook cells by prefixing them with `!`.
