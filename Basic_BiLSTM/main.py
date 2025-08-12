import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, 
                           f1_score, precision_recall_fscore_support)
from collections import Counter, defaultdict
import pickle
import re
import os
import json
import argparse
from typing import List, Tuple, Dict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Enhanced configuration with command line arguments support"""
    def __init__(self):
        # Model parameters
        self.VOCAB_SIZE = 10000
        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 128
        self.NUM_LAYERS = 2
        self.DROPOUT = 0.3
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 15
        self.SEQUENCE_LENGTH = 50
        
        # Data balancing
        self.BALANCE_METHOD = 'weighted_loss'
        self.MIN_SAMPLES_PER_CLASS = 1000
        
        # Punctuation mapping
        self.PUNCT_TO_ID = {
            'O': 0, ',': 1, '.': 2, '?': 3
        }
        self.ID_TO_PUNCT = {v: k for k, v in self.PUNCT_TO_ID.items()}
        
        # Paths
        self.DATA_PATH = 'data/'
        self.MODEL_PATH = 'models/'
        self.RESULTS_PATH = 'results/'
        
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
    
    def save_config(self, path: str):
        """Save configuration to JSON"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k != 'DEVICE'}
        config_dict['DEVICE'] = str(self.DEVICE)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class TextProcessor:
    """Enhanced text processor with better tokenization"""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq = Counter()
        self.char_to_remove = set('"""''„"«»')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        for char in self.char_to_remove:
            text = text.replace(char, '')
        
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts with progress tracking"""
        logger.info("Building vocabulary...")
        
        for text in tqdm(texts, desc="Processing texts for vocabulary"):
            cleaned_text = self.clean_text(text)
            words = self.tokenize(cleaned_text)
            self.word_freq.update(words)
        
        most_common = self.word_freq.most_common(self.vocab_size - 4)
        for word, freq in most_common:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")
        logger.info(f"Most common words: {dict(self.word_freq.most_common(10))}")
    
    def tokenize(self, text: str) -> List[str]:
        """Improved tokenization"""
        text = text.lower()
        text = re.sub(r'([,.;?!:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return [word for word in text.split() if word.strip()]
    
    def encode(self, words: List[str]) -> List[int]:
        """Convert words to indices"""
        return [self.word_to_idx.get(word, 1) for word in words]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to words"""
        return [self.idx_to_word.get(idx, '<UNK>') for idx in indices]

class DataBalancer:
    """Enhanced data balancer with better statistics"""
    def __init__(self, method: str = 'weighted_loss'):
        self.method = method
        self.balance_stats = {}
        
    def balance_data(self, sequences: List[List[int]], labels: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Balance dataset with detailed statistics"""
        logger.info(f"Balancing data using method: {self.method}")
        
        # Calculate original statistics
        flat_labels = [label for seq_labels in labels for label in seq_labels]
        original_counts = Counter(flat_labels)
        
        self.balance_stats['original'] = dict(original_counts)
        logger.info(f"Original label distribution: {original_counts}")
        
        if self.method == 'weighted_loss':
            return sequences, labels
        elif self.method == 'oversample':
            balanced_seq, balanced_labels = self._oversample(sequences, labels, original_counts)
        elif self.method == 'undersample':
            balanced_seq, balanced_labels = self._undersample(sequences, labels, original_counts)
        else:
            return sequences, labels
        
        # Calculate new statistics
        new_flat_labels = [label for seq_labels in balanced_labels for label in seq_labels]
        new_counts = Counter(new_flat_labels)
        self.balance_stats['balanced'] = dict(new_counts)
        logger.info(f"Balanced label distribution: {new_counts}")
        
        return balanced_seq, balanced_labels
    
    def _oversample(self, sequences, labels, label_counts):
        """Improved oversampling with better distribution"""
        max_count = max(label_counts.values())
        target_count = min(max_count // 3, 2000)  # Reasonable target
        
        # Group sequences by dominant punctuation
        seq_groups = defaultdict(list)
        for seq, seq_labels in zip(sequences, labels):
            punct_counts = Counter(seq_labels)
            # Skip sequences with only 'O' labels for minority classes
            non_o_labels = [l for l in seq_labels if l != 0]
            if non_o_labels:
                dominant_punct = max(Counter(non_o_labels), key=Counter(non_o_labels).get)
                seq_groups[dominant_punct].append((seq, seq_labels))
        
        balanced_sequences = []
        balanced_labels = []
        
        for punct_id, seq_pairs in seq_groups.items():
            current_count = len(seq_pairs)
            if current_count < target_count and punct_id != 0:  # Don't oversample 'O'
                # Oversample minority classes
                multiplier = target_count // current_count
                remainder = target_count % current_count
                
                for _ in range(multiplier):
                    for seq, seq_labels in seq_pairs:
                        balanced_sequences.append(seq)
                        balanced_labels.append(seq_labels)
                
                for seq, seq_labels in seq_pairs[:remainder]:
                    balanced_sequences.append(seq)
                    balanced_labels.append(seq_labels)
            else:
                for seq, seq_labels in seq_pairs:
                    balanced_sequences.append(seq)
                    balanced_labels.append(seq_labels)
        
        return balanced_sequences, balanced_labels
    
    def _undersample(self, sequences, labels, label_counts):
        """Improved undersampling"""
        minority_counts = [count for label, count in label_counts.items() if label != 0]
        if minority_counts:
            target_count = max(min(minority_counts) * 2, 1000)
        else:
            target_count = 1000
        
        seq_groups = defaultdict(list)
        for seq, seq_labels in zip(sequences, labels):
            punct_counts = Counter(seq_labels)
            dominant_punct = max(punct_counts, key=punct_counts.get)
            seq_groups[dominant_punct].append((seq, seq_labels))
        
        balanced_sequences = []
        balanced_labels = []
        
        for punct_id, seq_pairs in seq_groups.items():
            if punct_id == 0:  # Undersample 'O' class more aggressively
                selected = seq_pairs[:target_count // 2]
            else:
                selected = seq_pairs[:target_count]
            
            for seq, seq_labels in selected:
                balanced_sequences.append(seq)
                balanced_labels.append(seq_labels)
        
        return balanced_sequences, balanced_labels

class PunctuationDataset(Dataset):
    """Enhanced dataset with better padding and augmentation"""
    def __init__(self, sequences: List[List[int]], labels: List[List[int]], 
                 max_length: int = 50, augment: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        label = self.labels[idx].copy()
        
        # Data augmentation (simple word dropout)
        if self.augment and np.random.random() < 0.1:
            dropout_indices = np.random.choice(
                len(sequence), 
                size=max(1, len(sequence) // 10), 
                replace=False
            )
            for i in dropout_indices:
                if sequence[i] != 0:  # Don't dropout padding
                    sequence[i] = 1  # Replace with <UNK>
        
        # Pad or truncate
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            label = label[:self.max_length]
        else:
            pad_length = self.max_length - len(sequence)
            sequence = sequence + [0] * pad_length
            label = label + [0] * pad_length
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class FocalLoss(nn.Module):
    """Focal Loss implementation for handling imbalanced classes"""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BiLSTMPunctuationModel(nn.Module):
    """Enhanced Bi-LSTM model with attention mechanism"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.3):
        super(BiLSTMPunctuationModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = lstm_out * attention_weights
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        # Classification
        output = self.relu(self.fc1(attended_output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
class ModelEvaluator:
    """Comprehensive model evaluation with detailed reports"""
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model, data_loader, criterion, phase='test'):
        """Comprehensive model evaluation"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc=f"Evaluating {phase}")):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                output = model(data)
                
                # Calculate loss
                output_flat = output.view(-1, len(self.config.PUNCT_TO_ID))
                target_flat = target.view(-1)
                
                loss = criterion(output_flat, target_flat)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output_flat, dim=1)
                predictions = torch.argmax(output_flat, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target_flat.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        metrics['loss'] = avg_loss
        
        self.results[phase] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def _calculate_metrics(self, targets, predictions, probabilities):
        """Calculate comprehensive metrics"""
        # Filter out padding tokens
        valid_indices = [i for i, target in enumerate(targets) if target != 0]
        filtered_targets = [targets[i] for i in valid_indices]
        filtered_predictions = [predictions[i] for i in valid_indices]
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_targets, filtered_predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': sum(1 for t, p in zip(filtered_targets, filtered_predictions) if t == p) / len(filtered_targets),
            'macro_f1': f1_score(filtered_targets, filtered_predictions, average='macro', zero_division=0),
            'weighted_f1': f1_score(filtered_targets, filtered_predictions, average='weighted', zero_division=0),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist()
        }
        
        return metrics
    
    def generate_classification_report(self, phase='test'):
        """Generate detailed classification report"""
        if phase not in self.results:
            logger.error(f"No results available for phase: {phase}")
            return

        targets = self.results[phase]['targets']
        predictions = self.results[phase]['predictions']

    # Filter out padding tokens
        valid_indices = [i for i, target in enumerate(targets) if target != 0]
        filtered_targets = [targets[i] for i in valid_indices]
        filtered_predictions = [predictions[i] for i in valid_indices]

    # Generate classification report
        target_names = [self.config.ID_TO_PUNCT[i] for i in range(len(self.config.PUNCT_TO_ID))]
        labels = list(range(len(target_names)))  # Explicitly set labels to avoid mismatch

        report = classification_report(
            filtered_targets,
            filtered_predictions,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0
        )

    # Save report
        report_path = os.path.join(self.config.RESULTS_PATH, f'classification_report_{phase}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {phase.upper()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

        logger.info(f"Classification report saved to {report_path}")
        logger.info(f"\nClassification Report - {phase.upper()}:\n{report}")

        return report

    def plot_confusion_matrix(self, phase='test'):
        """Generate and save confusion matrix plot"""
        if phase not in self.results:
            logger.error(f"No results available for phase: {phase}")
            return
        
        targets = self.results[phase]['targets']
        predictions = self.results[phase]['predictions']
        
        # Filter out padding tokens
        valid_indices = [i for i, target in enumerate(targets) if target != 0]
        filtered_targets = [targets[i] for i in valid_indices]
        filtered_predictions = [predictions[i] for i in valid_indices]
        
        # Generate confusion matrix
        cm = confusion_matrix(filtered_targets, filtered_predictions)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[self.config.ID_TO_PUNCT[i] for i in range(len(self.config.PUNCT_TO_ID))],
                   yticklabels=[self.config.ID_TO_PUNCT[i] for i in range(len(self.config.PUNCT_TO_ID))])
        plt.title(f'Confusion Matrix - {phase.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = os.path.join(self.config.RESULTS_PATH, f'confusion_matrix_{phase}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {plot_path}")
    
    def plot_training_history(self, train_losses, val_losses, val_f1_scores):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot F1 scores
        ax2.plot(val_f1_scores, label='Validation F1', color='green')
        ax2.set_title('Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.config.RESULTS_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
    
    
    
    def save_detailed_results(self, phase='test'):
        """Save detailed results to file"""
        if phase not in self.results:
            logger.error(f"No results available for phase: {phase}")
            return
        
        results_path = os.path.join(self.config.RESULTS_PATH, f'detailed_results_{phase}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {
            'metrics': self.results[phase]['metrics'],
            'predictions': self.results[phase]['predictions'],
            'targets': self.results[phase]['targets']
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2,default=convert_numpy)
        
        logger.info(f"Detailed results saved to {results_path}")

class PunctuationPredictor:
    """Enhanced main predictor class"""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.processor = TextProcessor(self.config.VOCAB_SIZE)
        self.balancer = DataBalancer(self.config.BALANCE_METHOD)
        self.evaluator = ModelEvaluator(self.config)
        self.model = None
        self.class_weights = None
        self.writer = SummaryWriter(log_dir='runs/punctuation_experiment')
        
    def load_data_from_file(self, file_path: str) -> List[str]:
        """Load data from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
            logger.info(f"Loaded {len(texts)} texts from {file_path}")
            return texts
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
    
    def create_sample_data(self, save_to_file: bool = True):
        """Create sample data for testing"""
        sample_texts = [
            "Hello world, this is a test sentence. How are you today? I hope you are doing well; it has been a long time.",
            "The quick brown fox jumps over the lazy dog. This is another sentence, with some punctuation; and a question mark?",
            "Machine learning is fascinating, especially neural networks. Can you believe how far we have come? Yes, it is amazing; the future looks bright.",
            "Data science involves statistics, programming, and domain knowledge. What do you think about this field? It is growing rapidly; many opportunities exist.",
            "Natural language processing helps computers understand human language. This is important for many applications, including chatbots and translation systems; the technology keeps improving.",
            "Python is a versatile programming language. It is widely used in data science, web development, and automation. Do you enjoy coding? Programming can be both challenging and rewarding; practice makes perfect.",
            "The weather today is quite pleasant. It is sunny with a gentle breeze. Would you like to go for a walk? Fresh air is good for health; exercise is important too.",
            "Books are windows to different worlds. They transport us to new places and introduce us to fascinating characters. What is your favorite genre? Reading enhances vocabulary; knowledge is power.",
            "Technology continues to evolve rapidly. Smartphones, laptops, and tablets have become essential tools. How do you stay updated with tech trends? Innovation drives progress; adaptation is key.",
            "Cooking is both an art and a science. It requires creativity, precision, and patience. Do you enjoy trying new recipes? Food brings people together; sharing meals creates bonds."
        ]
        
        # Expand dataset
        expanded_texts = []
        for text in sample_texts:
            for _ in range(50):  # Create variations
                expanded_texts.append(text)
        
        if save_to_file:
            sample_data_path = os.path.join(self.config.DATA_PATH, 'sample_data.txt')
            with open(sample_data_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(expanded_texts))
            logger.info(f"Sample data saved to {sample_data_path}")
        
        return expanded_texts
    
    def prepare_data(self, texts: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
        """Enhanced data preparation with better statistics"""
        logger.info("Preparing data...")
        
        sequences = []
        labels = []
        skipped_texts = 0
        
        for text in tqdm(texts, desc="Processing texts"):
            try:
                cleaned_text = self.processor.clean_text(text)
                words, puncts = self._extract_words_and_punctuation(cleaned_text)
                
                if len(words) < 3:  # Skip very short texts
                    skipped_texts += 1
                    continue
                
                # Encode words
                word_ids = self.processor.encode(words)
                
                # Convert punctuation to labels
                punct_labels = [self.config.PUNCT_TO_ID.get(p, 0) for p in puncts]
                
                # Create overlapping sequences
                step_size = max(1, self.config.SEQUENCE_LENGTH // 3)
                for i in range(0, len(word_ids) - self.config.SEQUENCE_LENGTH + 1, step_size):
                    seq = word_ids[i:i + self.config.SEQUENCE_LENGTH]
                    seq_labels = punct_labels[i:i + self.config.SEQUENCE_LENGTH]
                    
                    if len(seq) == self.config.SEQUENCE_LENGTH:
                        sequences.append(seq)
                        labels.append(seq_labels)
                        
            except Exception as e:
                logger.warning(f"Error processing text: {str(e)}")
                skipped_texts += 1
        
        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Skipped {skipped_texts} texts due to errors or short length")
        
        return sequences, labels
    
    def _extract_words_and_punctuation(self, text: str) -> Tuple[List[str], List[str]]:
        """Enhanced word and punctuation extraction"""
        words = []
        puncts = []
        
        tokens = self.processor.tokenize(text)
        
        for token in tokens:
            if token in self.config.PUNCT_TO_ID:
                # This is a punctuation mark
                if words:  # Only add if we have words
                    puncts[-1] = token  # Replace the last 'O' with actual punctuation
            else:
                # This is a word
                words.append(token)
                puncts.append('O')  # Default to no punctuation
        
        # Ensure equal length
        while len(puncts) < len(words):
            puncts.append('O')
        
        return words, puncts
    
    def calculate_class_weights(self, labels: List[List[int]]) -> torch.Tensor:
        """Calculate class weights for imbalanced data - FIXED"""
        flat_labels = [label for seq_labels in labels for label in seq_labels]
        
        # Calculate weights - Convert to numpy array
        unique_labels = list(set(flat_labels))
        unique_labels_array = np.array(unique_labels)  # Convert to numpy array
        
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels_array,  # Use numpy array instead of list
            y=flat_labels
        )
        
        # Create weight tensor
        weight_tensor = torch.ones(len(self.config.PUNCT_TO_ID))
        for i, weight in zip(unique_labels, class_weights):
            weight_tensor[i] = weight
        
        logger.info(f"Class weights calculated: {weight_tensor.tolist()}")
        return weight_tensor
    
    def train_model(self, train_loader, val_loader, test_loader):
        """Enhanced training with better monitoring"""
        logger.info("Starting model training...")
        
        # Initialize model
        self.model = BiLSTMPunctuationModel(
            vocab_size=len(self.processor.word_to_idx),
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_classes=len(self.config.PUNCT_TO_ID),
            dropout=self.config.DROPOUT
        ).to(self.config.DEVICE)
        
        # Loss function and optimizer
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.config.DEVICE))
        else:
            criterion = FocalLoss(alpha=1, gamma=2)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        # Training tracking
        train_losses = []
        val_losses = []
        val_f1_scores = []
        best_f1 = 0.0
# Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            epoch_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Reshape for loss calculation
                output = output.view(-1, len(self.config.PUNCT_TO_ID))
                target = target.view(-1)
                
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
                
                # Update progress bar
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
                # Log to tensorboard
                self.writer.add_scalar('Training/Batch_Loss', loss.item(), 
                                     epoch * len(train_loader) + batch_idx)
            
            # Calculate average training loss
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluator.evaluate_model(self.model, val_loader, criterion, 'validation')
            val_loss = val_metrics['loss']
            val_f1 = val_metrics['weighted_f1']
            
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            
            # Learning rate scheduling
            scheduler.step(val_f1)
            
            # Log to tensorboard
            self.writer.add_scalar('Training/Epoch_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Validation/F1', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(self.config.MODEL_PATH, 'best_model.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'epoch': epoch,
                    'config': self.config.__dict__
                }, model_path)
                logger.info(f"New best model saved with F1: {val_f1:.4f}")
        
        # Final evaluation on test set
        test_metrics = self.evaluator.evaluate_model(self.model, test_loader, criterion, 'test')
        logger.info(f"Final Test Metrics: {test_metrics}")
        
        # Generate comprehensive evaluation
        self.evaluator.generate_classification_report('test')
        self.evaluator.plot_confusion_matrix('test')
        self.evaluator.plot_training_history(train_losses, val_losses, val_f1_scores)
        self.evaluator.save_detailed_results('test')
        
        self.writer.close()
        return test_metrics
    
    def predict_punctuation(self, text: str) -> str:
        """Enhanced prediction with better post-processing"""
        if self.model is None:
            logger.error("Model not trained yet!")
            return text
        
        self.model.eval()
        
        # Prepare input
        cleaned_text = self.processor.clean_text(text)
        words = self.processor.tokenize(cleaned_text)
        
        # Remove existing punctuation for prediction
        clean_words = [word for word in words if word not in self.config.PUNCT_TO_ID]
        
        if len(clean_words) == 0:
            return text
        
        # Encode words
        word_ids = self.processor.encode(clean_words)
        
        # Create sequences
        predicted_punctuation = ['O'] * len(clean_words)
        
        for i in range(0, len(word_ids), self.config.SEQUENCE_LENGTH // 2):
            seq = word_ids[i:i + self.config.SEQUENCE_LENGTH]
            
            # Pad sequence if needed
            if len(seq) < self.config.SEQUENCE_LENGTH:
                seq = seq + [0] * (self.config.SEQUENCE_LENGTH - len(seq))
            
            # Convert to tensor
            seq_tensor = torch.tensor([seq], dtype=torch.long).to(self.config.DEVICE)
            
            # Predict
            with torch.no_grad():
                output = self.model(seq_tensor)
                predictions = torch.argmax(output, dim=2).squeeze().cpu().numpy()
            
            # Extract predictions for this segment
            end_idx = min(i + self.config.SEQUENCE_LENGTH, len(clean_words))
            for j, pred in enumerate(predictions[:end_idx - i]):
                if i + j < len(predicted_punctuation):
                    predicted_punctuation[i + j] = self.config.ID_TO_PUNCT[pred]
        
        # Post-process predictions
        predicted_punctuation = self._post_process_predictions(predicted_punctuation)
        
        # Reconstruct text
        result = []
        for word, punct in zip(clean_words, predicted_punctuation):
            result.append(word)
            if punct != 'O':
                result.append(punct)
        
        return ' '.join(result)
    
    def _post_process_predictions(self, predictions: List[str]) -> List[str]:
        """Post-process predictions to improve quality"""
        processed = predictions.copy()
        
        # Rule 1: Don't put punctuation after very short words at the beginning
        for i in range(min(2, len(processed))):
            if len(processed[i]) <= 2:
                processed[i] = 'O'
        
        # Rule 2: Reduce consecutive punctuation
        for i in range(1, len(processed)):
            if processed[i] != 'O' and processed[i-1] != 'O':
                processed[i] = 'O'
        
        # Rule 3: Ensure sentence endings
        if len(processed) > 0 and processed[-1] == 'O':
            processed[-1] = '.'
        
        return processed
    
    def save_model_and_processor(self, model_path: str = None, processor_path: str = None):
        """Save model and processor"""
        if model_path is None:
            model_path = os.path.join(self.config.MODEL_PATH, 'final_model.pth')
        if processor_path is None:
            processor_path = os.path.join(self.config.MODEL_PATH, 'processor.pkl')
        
        # Save model
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'vocab_size': len(self.processor.word_to_idx)
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save processor
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        logger.info(f"Processor saved to {processor_path}")
    
    def load_model_and_processor(self, model_path: str = None, processor_path: str = None):
        """Load model and processor"""
        if model_path is None:
            model_path = os.path.join(self.config.MODEL_PATH, 'final_model.pth')
        if processor_path is None:
            processor_path = os.path.join(self.config.MODEL_PATH, 'processor.pkl')
        
        # Load processor
        try:
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
            logger.info(f"Processor loaded from {processor_path}")
        except FileNotFoundError:
            logger.error(f"Processor file not found: {processor_path}")
            return False
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.config.DEVICE)
            
            self.model = BiLSTMPunctuationModel(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=self.config.EMBEDDING_DIM,
                hidden_dim=self.config.HIDDEN_DIM,
                num_layers=self.config.NUM_LAYERS,
                num_classes=len(self.config.PUNCT_TO_ID),
                dropout=self.config.DROPOUT
            ).to(self.config.DEVICE)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return False
    
    def run_complete_pipeline(self, data_path: str = None):
        """Run complete training pipeline"""
        logger.info("Starting complete pipeline...")
        
        # Load or create data
        if data_path and os.path.exists(data_path):
            texts = self.load_data_from_file(data_path)
        else:
            logger.info("No data file provided, creating sample data...")
            texts = self.create_sample_data()
        
        if not texts:
            logger.error("No data available for training!")
            return
        
        # Build vocabulary
        self.processor.build_vocab(texts)
        
        # Prepare data
        sequences, labels = self.prepare_data(texts)
        
        if not sequences:
            logger.error("No sequences created from data!")
            return
        
        # Balance data
        sequences, labels = self.balancer.balance_data(sequences, labels)
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(labels)
        
        # Split data
        train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=0.3, random_state=42
        )
        
        val_sequences, test_sequences, val_labels, test_labels = train_test_split(
            temp_sequences, temp_labels, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = PunctuationDataset(train_sequences, train_labels, 
                                         self.config.SEQUENCE_LENGTH, augment=True)
        val_dataset = PunctuationDataset(val_sequences, val_labels, 
                                       self.config.SEQUENCE_LENGTH, augment=False)
        test_dataset = PunctuationDataset(test_sequences, test_labels, 
                                        self.config.SEQUENCE_LENGTH, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, 
                              shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, 
                               shuffle=False, num_workers=2)
        
        # Train model
        test_metrics = self.train_model(train_loader, val_loader, test_loader)
        
        # Save model and processor
        self.save_model_and_processor()
        
        # Save configuration
        config_path = os.path.join(self.config.RESULTS_PATH, 'config.json')
        self.config.save_config(config_path)
        
        logger.info("Pipeline completed successfully!")
        return test_metrics

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Bi-LSTM Punctuation Prediction')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict', 'test'],
                       help='Mode: train, predict, or test')
    parser.add_argument('--data_file', '--data_path', type=str, 
                       help='Path to training data file')
    parser.add_argument('--model_path', type=str, help='Path to save/load model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--balance_method', type=str, default='weighted_loss', 
                       choices=['weighted_loss', 'oversample', 'undersample'],
                       help='Data balancing method')
    parser.add_argument('--predict', type=str, help='Text to predict punctuation for')
    parser.add_argument('--load_model', action='store_true', help='Load existing model')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Create configuration
    config = Config()
    
    # Update config with command line arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.balance_method:
        config.BALANCE_METHOD = args.balance_method
    
    # Create predictor
    predictor = PunctuationPredictor(config)
    
    # Handle different modes
    if args.mode == 'predict' or args.predict:
        # Prediction mode
        if args.load_model or args.mode == 'predict':
            success = predictor.load_model_and_processor(args.model_path)
            if success:
                if args.predict:
                    result = predictor.predict_punctuation(args.predict)
                    print(f"Original: {args.predict}")
                    print(f"Predicted: {result}")
                else:
                    print("Please provide text to predict with --predict")
            else:
                print("Failed to load model!")
        else:
            print("Please use --load_model flag for prediction mode")
    elif args.mode == 'test':
        # Test mode
        logger.info("Running test mode...")
        test_example()
    else:
        # Training mode (default)
        logger.info("Starting training mode...")
        data_path = args.data_file if hasattr(args, 'data_file') and args.data_file else getattr(args, 'data_path', None)
        predictor.run_complete_pipeline(data_path)

if __name__ == "__main__":
    main()

# Example usage for testing
def test_example():
    """Test the punctuation predictor with sample data"""
    logger.info("Running test example...")
    
    # Create predictor
    predictor = PunctuationPredictor()
    
    # Run complete pipeline
    predictor.run_complete_pipeline()
    
    # Test prediction
    test_text = "Hello world this is a test sentence how are you today I hope you are doing well"
    predicted = predictor.predict_punctuation(test_text)
    
    print(f"Original: {test_text}")
    print(f"Predicted: {predicted}")
    
    logger.info("Test example completed!")

# Uncomment to run test
# test_example()
# ⚠️ Make sure your evaluator uses `labels` and `target_names` in classification_report:
# from sklearn.metrics import classification_report
# label_list = ['O', 'COMMA', 'PERIOD', 'QUESTION', 'EXCLAMATION', 'SEMICOLON']
# classification_report(..., labels=list(range(len(label_list))), target_names=label_list)
