import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define your Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, num_heads, num_encoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        out = self.fc(out[-1, :, :])  # Take the last output from the transformer
        return out

# Define your hybrid chatbot model
class HybridChatbot(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, transformer_model):
        super(HybridChatbot, self).__init__()
        self.lstm = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_hidden_size)
        self.transformer = transformer_model

    def forward(self, input_sequence):
        lstm_output = self.lstm(input_sequence)
        transformer_output = self.transformer(input_sequence)
        
        # Combine the outputs of LSTM and Transformer (you can customize how to combine)
        combined_output = lstm_output + transformer_output
        
        return combined_output
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
