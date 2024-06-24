import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DPRQuestionEncoder,
    DPRContextEncoder,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import faiss
import numpy as np
from tqdm import tqdm
import logging
import argparse
from torch.cuda.amp import GradScaler, autocast

class RAGDataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]

        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target = self.tokenizer.encode(
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': target.squeeze()
        }

class RAG(nn.Module):
    def __init__(self, retriever_model, generator_model, tokenizer, n_docs=5):
        super(RAG, self).__init__()
        self.retriever = retriever_model
        self.generator = generator_model
        self.tokenizer = tokenizer
        self.n_docs = n_docs

    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask, labels=None):
        # Retrieve relevant documents
        with torch.no_grad():
            retriever_outputs = self.retriever(input_ids, attention_mask)
            retrieved_doc_embeds = retriever_outputs.pooler_output

        # Generate answer
        generator_inputs = torch.cat([input_ids, context_input_ids], dim=1)
        generator_attention_mask = torch.cat([attention_mask, context_attention_mask], dim=1)

        if labels is not None:
            outputs = self.generator(
                input_ids=generator_inputs,
                attention_mask=generator_attention_mask,
                labels=labels
            )
        else:
            outputs = self.generator.generate(
                input_ids=generator_inputs,
                attention_mask=generator_attention_mask,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )

        return outputs, retrieved_doc_embeds

class RAGRetriever(nn.Module):
    def __init__(self, question_encoder, context_encoder, index, tokenizer):
        super(RAGRetriever, self).__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.index = index
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        question_embeddings = self.question_encoder(input_ids, attention_mask).pooler_output
        return question_embeddings

    def retrieve(self, question_embeddings, k=5):
        scores, indices = self.index.search(question_embeddings.cpu().numpy(), k)
        return scores, indices

def create_index(context_encoder, contexts, batch_size=32):
    index = faiss.IndexFlatIP(768)  # Assuming BERT base with 768 dimensions
    
    for i in tqdm(range(0, len(contexts), batch_size), desc="Creating index"):
        batch = contexts[i:i+batch_size]
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output.cpu().numpy()
        index.add(embeddings)
    
    return index

def train_rag(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device):
    scaler = GradScaler()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs, _ = model(input_ids, attention_mask, input_ids, attention_mask, labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Avg. Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs, _ = model(input_ids, attention_mask, input_ids, attention_mask, labels)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Avg. Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_rag_model.pth')
            logging.info("New best model saved!")
        
        scheduler.step()

def main(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    
    # Initialize tokenizers and models
    retriever_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    generator = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
    
    # Create index for retrieval
    contexts = train_dataset['context'] + val_dataset['context']
    index = create_index(context_encoder, contexts)
    
    # Initialize RAG model
    retriever = RAGRetriever(question_encoder, context_encoder, index, retriever_tokenizer)
    rag_model = RAG(retriever, generator, generator_tokenizer).to(device)
    
    # Prepare datasets
    train_data = RAGDataset(train_dataset['question'], train_dataset['context'], train_dataset['answers']['text'], generator_tokenizer)
    val_data = RAGDataset(val_dataset['question'], val_dataset['context'], val_dataset['answers']['text'], generator_tokenizer)
    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(rag_model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs
    )
    
    # Train the model
    train_rag(rag_model, train_dataloader, val_dataloader, optimizer, scheduler, args.num_epochs, device)
    
    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an advanced RAG model")
    parser.add_argument("--dataset_name", type=str, default="squad", help="Name of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the scheduler")
    
    args = parser.parse_args()
    main(args)