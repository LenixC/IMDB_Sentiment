import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

def train_model():
    # Load dataset
    full_dataset = load_dataset("stanfordnlp/imdb", split="train")
    dataset = full_dataset.shuffle(seed=42).select(range(10000))

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop
    num_epochs = 3  # Set the number of epochs
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
            labels = torch.tensor(batch["label"]).to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    model.save_pretrained("./model/")
    tokenizer.save_pretrained("./model/")

    # Test the model with sample sentences
    test_sentences = [
        "This movie was fantastic!",
        "I absolutely hated this film.",
        "It was just okay, not great.",
        "An absolute masterpiece!",
        "Waste of time!",
        "A beautiful story and well acted.",
        "Not my type of movie.",
        "It could have been better.",
        "A thrilling adventure from start to finish!",
        "Very disappointing."
    ]

    # Switch model to evaluation mode
    model.eval()

    # Prepare tokenizer for test inputs
    inputs = tokenizer(test_sentences, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Print predictions
    for sentence, prediction in zip(test_sentences, predictions):
        sentiment = "positive" if prediction.item() == 1 else "negative"
        print(f"Input: \"{sentence}\" -> Predicted sentiment: {sentiment}")

# Call the function to train the model and test it
train_model()
