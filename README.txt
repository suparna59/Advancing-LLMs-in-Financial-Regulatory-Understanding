

1. Project Overview
This project, "Advancing LLMs in Financial Regulatory Understanding," fine-tunes the LLaMA model to interpret and generate domain-specific content in the financial regulatory domain. Using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, the project optimizes the model's performance with minimal computational overhead. 

The system focuses on:
- Domain-specific fine-tuning using authoritative datasets (e.g., FDIC and SEC).
- Evaluation metrics such as ROUGE and BERTScore to measure syntactic and semantic accuracy.
- Real-world applicability in automating compliance workflows.


2. Key Contributions
- Domain-Specific Fine-Tuning: Adapts the LLaMA model to financial language using PEFT with LoRA.
- Data Augmentation and Preprocessing: Enhances model robustness through synonym replacement, tokenization, and normalization.
- Scalable Inference Pipeline: Optimized for real-time response generation.
- Evaluation Framework: Assesses performance using ROUGE and BERTScore metrics.

3. File Structure
# Main notebook for fine-tuning and evaluation
# List of project dependencies
# Directory for datasets
# Example financial regulatory dataset
# Dataset for abbreviations and definitions
# Project documentation

4. Requirements
Prerequisites
- Python 3.8 or newer
- CUDA-compatible GPU (for fine-tuning and inference)

Required Libraries
Install the dependencies with:
pip install -r 

Dependencies include:
- transformers
- torch
- datasets
- rouge_score
- bert-score
- peft
- trl
- nltk
- pandas
- numpy
- tqdm
- accelerate

5. Run the Model
i.Fine-Tuning the Model
To fine-tune the LLaMA model with your dataset:

Open the Notebook:

Use the Jupyter notebook Base_model_final_FINAL.ipynb to perform fine-tuning.

Ensure that the dataset is located in the specified path (e.g., data/financial_terms.csv).
Steps:

Load the base LLaMA model from Hugging Face.
Preprocess the dataset using tokenization and normalization.
Apply Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
Save the fine-tuned model to a specified directory.

Required Setup:

Ensure you have installed all dependencies from requirements.txt.
Use a CUDA-compatible GPU for efficient training.
Fine-Tuning Script Example (if using a Python script):

python run_finetune.py --dataset data/financial_terms.csv --output models/fine_tuned_llama

ii. Running the Inference Pipeline
To generate predictions using the fine-tuned model:

Run the Inference Script: Use the run_inference.py script to process input text and generate predictions:

python run_inference.py --input input.json --output output.json
Input File Format (input.json):

The input file should be a JSON array of text queries. Example:

  "What does 'State Bank' mean in financial regulations?",
  "Define CPA in the context of finance."

Output File Format (output.json):

The output file will contain model-generated responses. Example:

[
  "State Bank: A bank chartered by a state government.",
  "CPA: Certified Public Accountant."
]

Steps in the Script:

Load the fine-tuned model and tokenizer.
Process each query in the input file.
Save predictions to the output file.

iii. Interactive Notebook Usage
If you prefer an interactive approach:

Open the notebook:

jupyter notebook
Load Base_model_final_FINAL.ipynb.
Follow the steps to:
Load the fine-tuned model.
Enter input queries directly in the notebook.
View and evaluate the generated outputs.

iv. Notes and Best Practices
Default File Paths: Ensure datasets, models, and outputs are stored in the correct directories.
Error Handling: Common issues include missing dependencies, incorrect input formats, or GPU memory errors.
Example Outputs: Verify your results match expected examples like:
Input: "What does 'State Bank' mean?"
Output: "State Bank: A bank chartered by a state government."



6. Dataset
The project uses datasets sourced from:
- FDIC Terms Dataset: Structured definitions of financial terms.
- Financial Terminologies Dataset: Abbreviations and their full meanings.

Datasets are preprocessed using:
- Normalization
- Tokenization
- Semantic parsing
- Synonym replacement

 7. Evaluation Metrics
1. ROUGE: Measures n-gram overlap for syntactic accuracy.
2. BERTScore: Evaluates semantic similarity and alignment.

Example Evaluation Results:
- ROUGE-1: 0.75
- BERTScore: 0.82

8. Limitations
- Restricted to U.S.-based regulatory datasets.
- No multilingual support at present.
- Requires high-quality preprocessing for optimal results.

9. Future Work
- Expand to international regulatory datasets.
- Incorporate multilingual support.
- Enhance explainability and interpretability of outputs.

10. References
- Meta LLaMA: [Hugging Face](https://huggingface.co/meta-llama)
- ROUGE Metrics: [rouge_score](https://github.com/google-research/google-research/tree/master/rouge)
- LoRA Fine-Tuning: [LoRA Paper](https://arxiv.org/abs/2106.09685)
