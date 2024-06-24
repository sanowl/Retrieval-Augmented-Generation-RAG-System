# Retrieval-Augmented Generation (RAG) System

This project implements an advanced Retrieval-Augmented Generation (RAG) system, combining the power of dense retrieval and sequence-to-sequence generation for improved question-answering and text generation tasks.

## Features

- Dense Passage Retrieval (DPR) for efficient and effective document retrieval
- T5-based sequence-to-sequence model for flexible generation
- FAISS indexing for fast similarity search
- Mixed-precision training for improved performance and memory efficiency
- Dynamic batching and data handling
- Customizable hyperparameters via command-line arguments
- Comprehensive logging and model checkpointing

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.5+
- Datasets
- FAISS-gpu (or FAISS-cpu for CPU-only setups)
- NumPy
- tqdm

Install the required packages using:

```bash
pip install torch transformers datasets faiss-gpu numpy tqdm
```

## Usage

### Training

To train the RAG model, run the following command:

```bash
python rag_system.py \
    --dataset_name squad \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --warmup_steps 500
```

You can customize the training by modifying the command-line arguments:

- `--dataset_name`: Name of the dataset to use (default: "squad")
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_epochs`: Number of training epochs (default: 3)
- `--warmup_steps`: Number of warmup steps for the scheduler (default: 500)

### Inference

To use the trained RAG model for inference, you can use the following code snippet:

```python
import torch
from transformers import T5Tokenizer, DPRQuestionEncoder, DPRContextEncoder
from rag_system import RAG, RAGRetriever

def load_rag_model(model_path, device):
    # Initialize tokenizers and models
    retriever_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    generator = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
    
    # Load the trained RAG model
    retriever = RAGRetriever(question_encoder, context_encoder, None, retriever_tokenizer)
    rag_model = RAG(retriever, generator, generator_tokenizer)
    rag_model.load_state_dict(torch.load(model_path))
    rag_model.to(device)
    
    return rag_model, generator_tokenizer

def generate_answer(model, tokenizer, question, context, device):
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    outputs, _ = model(input_ids, attention_mask, input_ids, attention_mask)
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_answer

# Usage example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_rag_model.pth"
rag_model, tokenizer = load_rag_model(model_path, device)

question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital city is Paris, which is known for its iconic Eiffel Tower and world-class art museums like the Louvre."

answer = generate_answer(rag_model, tokenizer, question, context, device)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Model Architecture

The RAG system consists of three main components:

1. **Retriever**: Uses Dense Passage Retrieval (DPR) with BERT-based encoders for questions and contexts.
2. **Generator**: Employs a T5 sequence-to-sequence model for flexible answer generation.
3. **FAISS Index**: Enables efficient similarity search for retrieved documents.

The model combines these components to first retrieve relevant documents and then generate an answer based on the retrieved information and the original question.

## Training Process

The training process includes the following key steps:

1. Create a FAISS index for efficient document retrieval.
2. Encode questions and contexts using the DPR encoders.
3. Retrieve relevant documents based on the encoded questions.
4. Combine retrieved documents with the original question for input to the generator.
5. Generate answers using the T5 model.
6. Optimize the entire system end-to-end using a combination of retrieval and generation losses.

The training loop includes mixed-precision training for improved performance and memory efficiency, as well as learning rate scheduling with warm-up steps.

## Customization

You can customize various aspects of the RAG system:

1. **Dataset**: Change the `--dataset_name` argument to use different datasets compatible with the Hugging Face `datasets` library.
2. **Model Architecture**: Modify the `RAG` class to experiment with different retriever or generator architectures.
3. **Hyperparameters**: Adjust learning rate, batch size, number of epochs, and warm-up steps via command-line arguments.
4. **Loss Function**: Customize the loss calculation in the `train_rag` function to incorporate different objectives or weighting schemes.

## Performance Optimization

The current implementation includes several optimizations:

- Mixed-precision training using `torch.cuda.amp`
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warm-up
- FAISS indexing for fast similarity search

Further optimizations could include:

- Distributed training across multiple GPUs
- Quantization for reduced model size and inference speed
- Caching of retrieved documents to reduce computational overhead

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed and up to date.
2. Check that you have sufficient GPU memory for the chosen batch size.
3. For out-of-memory errors, try reducing the batch size or using gradient accumulation.
4. If using a custom dataset, ensure it's in the correct format and properly loaded.

## Contributing

Contributions to improve the RAG system are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- The RAG model is based on the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al.
- We use the Hugging Face Transformers library for pre-trained models and the Datasets library for data handling.
- The FAISS library by Facebook AI Research is used for efficient similarity search.