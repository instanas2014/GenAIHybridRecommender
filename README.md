# GenAIHybridRecommender

A hybrid recommendation system demo that showcases how Generative AI can augment deep collaborative filtering models with recent client-specific information.

##  Overview

This project demonstrates an approach to enhance traditional Deep Collaborative Filtering (Deep CF) recommendation systems by incorporating real-time user context and feedback via Generative AI (e.g., OpenAI's GPT models). It includes a Streamlit-based user interface and runs in a containerized Jupyter environment for ease of experimentation.

##  Project Structure

```
GenAIHybridRecommender/
│
├── docker-compose.yml            # For spinning up Jupyter (PyTorch) stack
├── environment.yml               # Conda environment for Streamlit and dependencies
├── Notebooks/                    # Code is here                   
└── README.md                     # You're here!
```

## 🧰 Prerequisites

### 1. Jupyter Stack (Deep Learning Environment)

You can use the provided `docker-compose.yml` file to start a JupyterLab environment (PyTorch-based):

```bash
docker-compose up -d
```

> ⚠️ Ensure Docker is installed and running on your machine.

### 2. Conda Environment in local coputer (for Streamlit UI)
You need to setup 
Install the required Python packages and create a dedicated environment using:

```bash
conda env create -f environment.yml
```

Set your OpenAI API key (required for GenAI augmentation):

```bash
conda env config vars set OPEN_AI_KEY="your_openai_api_key_here"
```

> Replace `"your_openai_api_key_here"` with your actual OpenAI API key.


## 🚀 Running the Application

Navigate to the `Notebooks` directory and launch the Streamlit app:

```bash
cd Notebooks
streamlit run RecommenderUI.py
```

This will open a browser window where you can interact with the hybrid recommender system.

## 📌 Notes

* This is a demo project and not optimized for production use.
* Make sure your OpenAI API key has sufficient quota and permissions.



