import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import AdamW
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tokenizers import ByteLevelBPETokenizer, Tokenizer
import logging

# Set up logging
logging.basicConfig(filename='shakespeare_predictor.log', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    print("Using CUDA GPU for training")
    # Set memory growth to avoid memory allocation errors
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU for training")

# Constants
DATA_DIR = "Data/"
MODEL_PATH = "shakespeare_bpe_model.keras"
TOKENIZER_PATH = "shakespeare_bpe_tokenizer"
SEQUENCE_LENGTH = 20  # Increased from 15 to account for subword tokens
BATCH_SIZE = 128
EPOCHS = 300  # Reduced slightly for faster training
EMBEDDING_DIM = 256  # Increased from 200 for better representation
LEARNING_RATE = 3e-4
VOCAB_SIZE = 20000  # Increased from 15000 to capture more subword units
SEED = 42

def load_data():
    """Load and combine data from CSV and TXT files in the Data folder."""
    print("Loading data...")
    logging.info("Loading data from Data directory")
    text_data = []
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
        logging.info(f"Created data directory: {DATA_DIR}")
        print("Please place your Shakespeare text files in this directory")
        return ""
    
    # Load TXT files
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} TXT files")
    for file in txt_files:
        try:
            with open(os.path.join(DATA_DIR, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                text_data.extend(lines)
                print(f"Loaded {len(lines)} lines from {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            logging.error(f"Error loading {file}: {str(e)}")
    
    # Load CSV files
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            if 'PlayerLine' in df.columns:
                lines = df['PlayerLine'].dropna().tolist()
                text_data.extend(lines)
                print(f"Loaded {len(lines)} lines from {file} (PlayerLine column)")
            else:
                print(f"Warning: No 'PlayerLine' column found in {file}")
                # Try to find alternative text columns
                text_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'line', 'dialogue', 'speech'])]
                if text_cols:
                    selected_col = text_cols[0]
                    lines = df[selected_col].dropna().tolist()
                    text_data.extend(lines)
                    print(f"Using alternative column: {selected_col}, loaded {len(lines)} lines from {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            logging.error(f"Error loading {file}: {str(e)}")
    
    # Clean the text data
    cleaned_text = []
    for line in text_data:
        if isinstance(line, str) and len(line.strip()) > 0:
            # Only remove control characters and keep all punctuation/symbols relevant to Shakespeare
            clean_line = re.sub(r'[\x00-\x1F\x7F]', '', line)
            # Normalize whitespace
            clean_line = re.sub(r'\s+', ' ', clean_line).strip()
            cleaned_text.append(clean_line)
    
    print(f"Loaded {len(cleaned_text)} lines of text")
    logging.info(f"Loaded {len(cleaned_text)} lines of text")
    
    if len(cleaned_text) == 0:
        print("WARNING: No text data was loaded. Please check your data files.")
        logging.warning("No text data was loaded. Please check data files.")
        return ""
        
    return ' '.join(cleaned_text)

def preprocess_data(text):
    """Tokenize the text using BPE and create sequences for training."""
    print("Preprocessing data with BPE tokenizer...")
    logging.info("Preprocessing data with BPE tokenizer")
    
    # Write the text to a file for tokenizer training
    temp_file = "shakespeare_text.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Text file created: {temp_file} ({len(text)} characters)")
    
    # Initialize and train the BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train it on Shakespeare data with appropriate parameters
    try:
        tokenizer.train(
            files=[temp_file],
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        )
        print(f"Tokenizer trained with vocab size {VOCAB_SIZE}")
    except Exception as e:
        print(f"Error training tokenizer: {str(e)}")
        logging.error(f"Error training tokenizer: {str(e)}")
        raise
    
    # Save the tokenizer
    os.makedirs(TOKENIZER_PATH, exist_ok=True)
    tokenizer.save_model(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    
    # Now encode the entire text to get token IDs
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    print(f"Text encoded into {len(tokens)} tokens")
    
    # Create input sequences and target tokens
    X = []
    y = []
    
    print("Creating training sequences...")
    for i in range(len(tokens) - SEQUENCE_LENGTH):
        if i % 100000 == 0:
            print(f"Processed {i}/{len(tokens) - SEQUENCE_LENGTH} tokens")
        
        input_sequence = tokens[i:i+SEQUENCE_LENGTH]
        target_token = tokens[i+SEQUENCE_LENGTH]
        
        X.append(input_sequence)
        y.append(target_token)
    
    print(f"Created {len(X)} sequence pairs")
    
    # Convert lists to numpy arrays
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Get the vocabulary size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocabulary size: {actual_vocab_size}")
    
    # Clean up temporary file
    try:
        os.remove(temp_file)
        print(f"Cleaned up temporary file: {temp_file}")
    except:
        pass
    
    return X, y, actual_vocab_size, tokenizer

def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * tf.keras.activations.sigmoid(x)

def build_model(vocab_size, embedding_dim=256):
    """Build the bidirectional LSTM model."""
    print("Building Bidirectional LSTM model...")
    logging.info("Building Bidirectional LSTM model")
    
    # Define input layer
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH,))
    
    # Embedding layer
    x = Embedding(vocab_size, embedding_dim)(inputs)
    
    # First Bidirectional LSTM layer
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_initializer='glorot_uniform'))(x)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(128, recurrent_initializer='glorot_uniform'))(x)
    x = Dropout(0.3)(x)
    
    # SwiGLU block instead of dense ReLU
    gate = Dense(256)(x)
    gate = swish(gate)
    linear = Dense(256)(x)
    x = gate * linear
    
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use AdamW optimizer with weight decay
    optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def train_model(X, y, vocab_size):
    """Train the LSTM model."""
    print("Training the model...")
    logging.info("Starting model training")
    
    model = build_model(vocab_size, EMBEDDING_DIM)
    
    # Set up model checkpoint to save best model
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,
        verbose=1,
        restore_best_weights=True
    )
    
    # Add learning rate reduction callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Split the data into training and validation sets
    indices = np.arange(len(X))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    
    split = int(0.9 * len(X))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    
    # Start training
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Model training completed!")
    logging.info("Model training completed")
    
    # Save training history for analysis
    with open('training_history.json', 'w') as f:
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]  # Convert numpy values to Python floats
        json.dump(history_dict, f)
    
    print("Training history saved to training_history.json")
    
    return model

def load_bpe_tokenizer():
    """Load the BPE tokenizer from the saved files."""
    vocab_file = os.path.join(TOKENIZER_PATH, "vocab.json")
    merges_file = os.path.join(TOKENIZER_PATH, "merges.txt")
    
    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        try:
            tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
            print(f"Tokenizer loaded from {TOKENIZER_PATH}")
            return tokenizer
        except Exception as e:
            raise FileNotFoundError(f"Error loading tokenizer: {str(e)}")
    else:
        raise FileNotFoundError(f"Tokenizer files not found in {TOKENIZER_PATH}")

def predict_next_words(model, tokenizer, text, num_words=5, temperature=0.2):
    """Predict the next complete words (not just tokens) given a sequence of text."""
    # First, encode the text to get token IDs
    encoded = tokenizer.encode(text)
    
    # Get the sequence
    sequence = encoded.ids
    
    # If sequence is longer than SEQUENCE_LENGTH, keep only the last SEQUENCE_LENGTH tokens
    if len(sequence) > SEQUENCE_LENGTH:
        sequence = sequence[-SEQUENCE_LENGTH:]
    # If sequence is shorter than SEQUENCE_LENGTH, pad with zeros (assuming 0 is the padding index)
    elif len(sequence) < SEQUENCE_LENGTH:
        sequence = [0] * (SEQUENCE_LENGTH - len(sequence)) + sequence
    
    # Convert to numpy array and reshape for prediction
    sequence = np.array(sequence).reshape(1, -1)
    sequence = np.array(sequence, dtype=np.int32)
    
    # Get predictions
    predictions = model.predict(sequence, verbose=0)[0]
    
    # Apply temperature to predictions
    if temperature != 1.0:
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
    
    # Get top predictions
    top_indices = predictions.argsort()[-50:][::-1]  # Get top 50 to filter down to words
    
    # Maintain a set of whole words we've seen to avoid duplicates
    predicted_words = []
    predicted_tokens_set = set()
    
    # Get the current context to help with word boundary detection
    current_context = text.strip()
    
    # Check if we're in the middle of a word or at a word boundary
    is_at_word_boundary = current_context == "" or current_context[-1].isspace() or not current_context[-1].isalnum()
    
    # Define punctuation characters to filter out
    punctuation_chars = set(".,!?;:'\"()[]{}…–—-")
    
    for idx in top_indices:
        # Get the token
        token = tokenizer.decode([idx]).strip()
        
        # Skip empty tokens
        if not token:
            continue
            
        # Make sure token contains at least one alphanumeric character
        if not any(c.isalnum() for c in token):
            continue
            
        # If we're in the middle of a word, only consider tokens that continue the word
        if not is_at_word_boundary and (token.startswith(" ") or token in punctuation_chars): 
            continue
            
        # If we're at word boundary, skip tokens that are just spaces or punctuation
        if is_at_word_boundary and not any(c.isalnum() for c in token):
            continue
        
        # Add the token if we haven't seen it before
        if token not in predicted_tokens_set:
            predicted_tokens_set.add(token)
            predicted_words.append(token)
            
        # Stop once we have enough predictions
        if len(predicted_words) >= num_words:
            break
    
    # If we still need more words, return what we have
    return predicted_words[:num_words]

class WordCompletionApp:
    def __init__(self, root, model, tokenizer):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        
        self.root.title("Shakespeare Word Predictor")
        self.root.geometry("800x600")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("Selected.TButton", background="#4CAF50", foreground="white")
        
        # Main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input text area
        input_frame = ttk.LabelFrame(main_frame, text="Enter your text (Shakespeare style):")
        input_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, width=70, height=10, wrap=tk.WORD, font=("Georgia", 11))
        self.text_input.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.text_input.bind("<KeyRelease>", self.on_key_release)
        self.text_input.focus_set()
        
        # Add keyboard shortcuts 
        self.text_input.bind("<Alt-1>", lambda e: self.use_suggestion_by_index(0))
        self.text_input.bind("<Alt-2>", lambda e: self.use_suggestion_by_index(1))
        self.text_input.bind("<Alt-3>", lambda e: self.use_suggestion_by_index(2))
        self.text_input.bind("<Alt-4>", lambda e: self.use_suggestion_by_index(3))
        self.text_input.bind("<Alt-5>", lambda e: self.use_suggestion_by_index(4))
        
        # Temperature control
        temp_frame = ttk.Frame(main_frame)
        temp_frame.pack(pady=5, padx=10, fill=tk.X)
        
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT, padx=5)
        self.temperature = tk.DoubleVar(value=0.6)
        temp_scale = ttk.Scale(temp_frame, from_=0.1, to=1.0, variable=self.temperature, 
                              orient=tk.HORIZONTAL, length=200, command=self.on_temp_change)
        temp_scale.pack(side=tk.LEFT, padx=5)
        
        self.temp_label = ttk.Label(temp_frame, text="0.60")
        self.temp_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(temp_frame, text="(Lower = more predictable, Higher = more creative)").pack(side=tk.LEFT, padx=5)
        
        # Suggestion frame
        self.suggestion_frame = ttk.LabelFrame(main_frame, text="Next Word Suggestions (Alt+1 to Alt+5)")
        self.suggestion_frame.pack(pady=10, padx=10, fill=tk.X)
        
        suggestion_inner_frame = ttk.Frame(self.suggestion_frame)
        suggestion_inner_frame.pack(pady=5, padx=5, fill=tk.X)
        
        self.suggestions = []
        for i in range(5):
            btn = ttk.Button(suggestion_inner_frame, text="", width=15)
            btn.grid(row=0, column=i, padx=5, pady=5)
            btn.configure(command=lambda b=btn, idx=i: self.use_suggestion_by_index(idx))
            self.suggestions.append(btn)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=5, padx=10, fill=tk.X)
        
        clear_btn = ttk.Button(control_frame, text="Clear Text", command=self.clear_text)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        add_period_btn = ttk.Button(control_frame, text="Add Period", command=lambda: self.add_punctuation("."))
        add_period_btn.pack(side=tk.LEFT, padx=5)
        
        add_comma_btn = ttk.Button(control_frame, text="Add Comma", command=lambda: self.add_punctuation(","))
        add_comma_btn.pack(side=tk.LEFT, padx=5)
        
        add_question_btn = ttk.Button(control_frame, text="Add Question Mark", command=lambda: self.add_punctuation("?"))
        add_question_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_temp_change(self, value):
        # Update the label when the temperature slider changes
        self.temp_label.configure(text=f"{float(value):.2f}")
        # Trigger new predictions with the new temperature
        self.on_key_release(None)
    
    def on_key_release(self, event):
        # Get current text
        current_text = self.text_input.get("1.0", tk.END).strip()
        
        if not current_text:
            for btn in self.suggestions:
                btn.configure(text="")
            return
        
        self.status.configure(text="Predicting...")
        self.root.update_idletasks()  # Force UI update
        
        try:
            # Predict next words with current temperature
            next_words = predict_next_words(
                self.model, 
                self.tokenizer, 
                current_text, 
                num_words=5, 
                temperature=self.temperature.get()
            )
            
            # Update suggestion buttons
            for i, word in enumerate(next_words):
                if i < len(self.suggestions):
                    # Add hotkey hints to display
                    display_text = f"{word}"
                    self.suggestions[i].configure(text=display_text)
            
            # Clear any remaining buttons
            for i in range(len(next_words), len(self.suggestions)):
                self.suggestions[i].configure(text="")
                
            self.status.configure(text="Ready")
        except Exception as e:
            self.status.configure(text=f"Error: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")
    
    def use_suggestion_by_index(self, index):
        if index < len(self.suggestions):
            word = self.suggestions[index].cget("text")
            self.use_suggestion(word)
    
    def use_suggestion(self, word):
        if not word:
            return
        
        # Get current text and cursor position
        current_text = self.text_input.get("1.0", tk.END)
        cursor_pos = self.text_input.index(tk.INSERT)
        
        # Add space before word if needed
        if current_text and current_text.strip() and not current_text.rstrip().endswith(" "):
            word = " " + word
        
        # Insert the word at cursor position
        self.text_input.insert(cursor_pos, word + " ")
        
        # Set focus back to text input
        self.text_input.focus_set()
        
        # Trigger prediction for the next word
        self.on_key_release(None)
    
    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        for btn in self.suggestions:
            btn.configure(text="")
        self.text_input.focus_set()
    
    def add_punctuation(self, punct):
        # Get current position
        cursor_pos = self.text_input.index(tk.INSERT)
        
        # Insert punctuation
        self.text_input.insert(cursor_pos, punct + " ")
        
        # Set focus back to text input
        self.text_input.focus_set()
        
        # Trigger new predictions
        self.on_key_release(None)

def main():
    try:
        # Set up logging
        logging.info("Application started")
        
        # Check if model and tokenizer already exist
        model_exists = os.path.exists(MODEL_PATH)
        tokenizer_exists = os.path.exists(os.path.join(TOKENIZER_PATH, "vocab.json")) and os.path.exists(os.path.join(TOKENIZER_PATH, "merges.txt"))
        
        if model_exists and tokenizer_exists:
            print(f"Loading existing model from {MODEL_PATH}")
            logging.info(f"Loading existing model from {MODEL_PATH}")
            
            try:
                model = load_model(MODEL_PATH, custom_objects={"swish": swish})
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                print(error_msg)
                logging.error(error_msg)
                raise
            
            print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
            logging.info(f"Loading existing tokenizer from {TOKENIZER_PATH}")
            
            try:
                tokenizer = load_bpe_tokenizer()
            except Exception as e:
                error_msg = f"Error loading tokenizer: {str(e)}"
                print(error_msg)
                logging.error(error_msg)
                raise
        else:
            print("No existing model or tokenizer found. Training a new model...")
            logging.info("Training new model and tokenizer")
            
            text = load_data()
            if not text:
                error_msg = "No text data available for training. Please add data files to the Data directory."
                print(error_msg)
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            X, y, vocab_size, tokenizer = preprocess_data(text)
            model = train_model(X, y, vocab_size)
        
        # Start the GUI
        print("Starting the Shakespeare word completion application...")
        logging.info("Starting application GUI")
        
        # Set up the root window
        root = tk.Tk()
        root.title("Shakespeare Word Predictor")
        
        # Add a window icon if available
        try:
            root.iconbitmap("shakespeare.ico")
        except:
            pass
        
        # Configure the application theme
        style = ttk.Style()
        try:
            style.theme_use('clam')  # Use a more modern theme if available
        except:
            pass
        
        # Create and run the application
        app = WordCompletionApp(root, model, tokenizer)
        
        # Center the window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'+{x}+{y}')
        
        root.mainloop()
        
    except Exception as e:
        error_message = f"ERROR: {str(e)}"
        print(error_message)
        logging.error(error_message, exc_info=True)
        
        # Show error dialog
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror("Shakespeare Word Predictor Error", 
                              f"An error occurred:\n\n{str(e)}\n\nCheck shakespeare_predictor.log for details.")
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()