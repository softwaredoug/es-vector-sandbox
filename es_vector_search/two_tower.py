import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from es_vector_search.wands_data import WANDSDataset


class TwoTowerModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.embedding_dim = embedding_dim

        # Optional: Projection layers to reduce embedding size
        self.doc_proj = nn.Linear(embedding_dim, embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

        # Document feature projections
        self.product_name_proj = nn.Linear(embedding_dim, embedding_dim)
        self.product_description_proj = nn.Linear(embedding_dim, embedding_dim)

    def encode_text(self, encoded):
        output = self.text_encoder(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        # Use CLS token representation
        return output.last_hidden_state[:, 0, :]

    def forward(self, query_tokens, product_token_features):
        query_emb = self.encode_text(query_tokens)
        query_emb = self.query_proj(query_emb)

        # Product name and description
        doc_features = []
        name_embedding = self.encode_text(product_token_features['product_name'])
        name_embedding = self.product_name_proj(name_embedding)
        doc_features.append(name_embedding)

        description_embedding = self.encode_text(product_token_features['product_description'])
        description_embedding = self.product_description_proj(description_embedding)
        doc_features.append(description_embedding)

        # Concatenate product name and description embeddings
        doc_emb = torch.stack(doc_features, dim=0).mean(dim=0)
        doc_emb = self.doc_proj(doc_emb)

        # Normalize embeddings (optional, helps with cosine similarity)
        query_emb = nn.functional.normalize(query_emb, dim=1)
        doc_emb = nn.functional.normalize(doc_emb, dim=1)

        return query_emb, doc_emb


def contrastive_loss(query_emb, doc_emb, temperature=0.05):
    scores = torch.matmul(query_emb, doc_emb.t()) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


def train(start_epoch=0, epochs=3):

    # cuda then mps then cpu
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize dataset and dataloader
    dataset = WANDSDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model and optimizer
    model = TwoTowerModel().to(device)
    if start_epoch > 0:
        model = TwoTowerModel().to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            queries = list(batch[0])
            product_names = list(batch[1])
            product_descriptions = list(batch[2])

            query_tokens = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
            product_name_tokens = tokenizer(product_names, padding=True, truncation=True, return_tensors="pt").to(device)
            product_description_tokens = tokenizer(product_descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
            product_text_tokens = {
                'product_name': product_name_tokens,
                'product_description': product_description_tokens
            }

            optimizer.zero_grad()
            query_emb, doc_emb = model(query_tokens, product_text_tokens)
            loss = contrastive_loss(query_emb, doc_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Cleanup
            del query_tokens, product_text_tokens, query_emb, doc_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"two_tower_epoch_{epoch + 1}.pth")
            print(f"Checkpoint saved for epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    train(epochs=3)
