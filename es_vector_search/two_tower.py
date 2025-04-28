import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from es_vector_search.wands_data import WandsDataset


class TwoTowerModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Optional: Projection layers to reduce embedding size
        self.doc_proj = nn.Linear(embedding_dim, embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

    def encode_text(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output = self.text_encoder(**encoded)
        # Use CLS token representation
        return output.last_hidden_state[:, 0, :]

    def forward(self, query_texts, product_names, product_descriptions):
        # Encode queries
        query_emb = self.encode_text(query_texts)
        query_emb = self.query_proj(query_emb)

        # Concatenate product name and description
        product_texts = [f"{name} {desc}" for name, desc in zip(product_names, product_descriptions)]
        doc_emb = self.encode_text(product_texts)
        doc_emb = self.doc_proj(doc_emb)

        # Normalize embeddings (optional, helps with cosine similarity)
        query_emb = nn.functional.normalize(query_emb, dim=1)
        doc_emb = nn.functional.normalize(doc_emb, dim=1)

        return query_emb, doc_emb


def contrastive_loss(query_emb, doc_emb, temperature=0.05):
    scores = torch.matmul(query_emb, doc_emb.t()) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


def train(epochs=3):

    # cuda then mps then cpu
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize dataset and dataloader
    dataset = WANDSDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model and optimizer
    model = TwoTowerModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            query_texts = batch['query'].to(device)
            product_names = batch['product_name'].to(device)
            product_descriptions = batch['product_description'].to(device)

            optimizer.zero_grad()
            query_emb, doc_emb = model(query_texts, product_names, product_descriptions)
            loss = contrastive_loss(query_emb, doc_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    train(epochs=3)

