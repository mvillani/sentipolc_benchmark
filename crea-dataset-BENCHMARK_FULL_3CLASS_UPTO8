#================================================================================
#GENERATORE DATASET AUMENTATO FEW-SHOT 3-CLASS (CON RANDOM VARIABILE PER RIGA)
#================================================================================
#Genera il file 'test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8.csv' contenente:
#- 8 Esempi Random Fissi (stessi esempi per tutti i tweet)
#- 8 Esempi Random Variabili (8 tweet casuali estratti al volo per ogni riga)
#- 8 Esempi Semantici MiniLM (top-8 per Cosine Similarity)
#
#Tutte le risposte sono etichettate come Positivo, Negativo o Neutro.
#================================================================================
#

import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import drive

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Libreria sentence-transformers non installata. Esegui: pip install -q sentence-transformers")
    sys.exit(1)

# ==============================================================================
# 0. MONTAGGIO DRIVE E CONFIGURAZIONE PERCORSI
# ==============================================================================

drive.mount("/content/drive", force_remount=False)

DRIVE_DIR = "/content/drive/MyDrive/sentipolc_eval"

# ==============================================================================
# 1. PERCORSI FILE & CONFIGURAZIONE
# ==============================================================================
TRAIN_DATASET_PATH = '/content/drive/MyDrive/sentipolc_eval/training_set_sentipolc16.csv'
TEST_DATASET_PATH = '/content/drive/MyDrive/sentipolc_eval/test_set_sentipolc16_gold2000.csv'
OUTPUT_DATASET_PATH = '/content/drive/MyDrive/sentipolc_eval/test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8.csv'

MODEL_EMBEDDING_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_K_SHOTS = 8

# ==============================================================================
# 2. FUNZIONI DI FORMATTAZIONE E MAPPATURA
# ==============================================================================
def map_sentiment_to_3class(opos: int, oneg: int) -> str:
    """Mappa le polarità binarie in Positivo, Negativo, Neutro. Scarta i casi misti."""
    op, on = int(opos), int(oneg)
    if op == 1 and on == 0:
        return "Positivo"
    elif op == 0 and on == 1:
        return "Negativo"
    elif op == 0 and on == 0:
        return "Neutro"
    else:
        return None

def format_fewshot_context(examples: list) -> str:
    """Formatta la lista di esempi nel blocco di testo Few-Shot."""
    formatted_blocks = []
    for idx, ex in enumerate(examples, 1):
        formatted_blocks.append(f'Esempio {idx}:\nTweet: "{ex["text"]}"\nRisposta: {ex["label"]}')
    return "\n\n".join(formatted_blocks)

# ==============================================================================
# 3. CARICAMENTO DATI
# ==============================================================================
def generate_augmented_dataset():
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(TEST_DATASET_PATH):
        print(f"❌ ERRORE: Assicurati che '{TRAIN_DATASET_PATH}' e '{TEST_DATASET_PATH}' siano presenti.")
        return

    print("📖 Caricamento e pulizia Training Set...")
    df_train_raw = pd.read_csv(TRAIN_DATASET_PATH)
    train_clean_rows = []
    for _, row in df_train_raw.iterrows():
        label = map_sentiment_to_3class(row['opos'], row['oneg'])
        if label is not None:
            text = str(row['text']).strip().replace('\n', ' ')
            train_clean_rows.append({'text': text, 'label': label})
    df_train = pd.DataFrame(train_clean_rows)

    print("📖 Caricamento Test Set...")
    rows_test = []
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 8)
            if len(parts) >= 9:
                tweet_id = parts[0].strip('"')
                opos, oneg = parts[2].strip('"'), parts[3].strip('"')
                text = parts[8].strip().strip('"')
                if opos.isdigit() and oneg.isdigit():
                    rows_test.append({
                        'id': tweet_id,
                        'target_opos': int(opos),
                        'target_oneg': int(oneg),
                        'text': text
                    })
    df_test = pd.DataFrame(rows_test)

    # --------------------------------------------------------------------------
    # A) FEW-SHOT RANDOM FISSO (Stessi 8 esempi per tutto il test set)
    # --------------------------------------------------------------------------
    print("\n🎲 1. Generazione Few-Shot Casuale FISSO (8-Shot)...")
    random.seed(42)
    fixed_indices = random.sample(range(len(df_train)), MAX_K_SHOTS)
    fixed_examples = [
        {'text': df_train.iloc[idx]['text'], 'label': df_train.iloc[idx]['label']}
        for idx in fixed_indices
    ]
    df_test['fewshot_random_3class_context'] = format_fewshot_context(fixed_examples)

    # --------------------------------------------------------------------------
    # B) FEW-SHOT RANDOM VARIABILE (8 esempi casuali diversi per OGNI riga)
    # --------------------------------------------------------------------------
    print("🎲 2. Generazione Few-Shot Casuale VARIABILE per singola riga...")
    variable_random_contexts = []
    for idx, row in df_test.iterrows():
        # Utilizziamo l'hash dell'ID tweet come seed per garantire riproducibilità
        seed_val = int(str(row['id'])[-6:]) if str(row['id']).isdigit() else idx
        random.seed(seed_val)

        row_indices = random.sample(range(len(df_train)), MAX_K_SHOTS)
        row_examples = [
            {'text': df_train.iloc[i]['text'], 'label': df_train.iloc[i]['label']}
            for i in row_indices
        ]
        variable_random_contexts.append(format_fewshot_context(row_examples))

    df_test['fewshot_random_variable_3class_context'] = variable_random_contexts

    # --------------------------------------------------------------------------
    # C) FEW-SHOT SEMANTICO (Top-8 Cosine Similarity con MiniLM)
    # --------------------------------------------------------------------------
    print(f"\n🧠 3. Caricamento SentenceTransformer [{MODEL_EMBEDDING_NAME}]...")
    embedder = SentenceTransformer(MODEL_EMBEDDING_NAME)

    print("⚡ Calcolo embedding per Training Set e Test Set...")
    train_vecs = embedder.encode(df_train['text'].tolist(), show_progress_bar=False, batch_size=64)
    test_vecs = embedder.encode(df_test['text'].tolist(), show_progress_bar=False, batch_size=64)

    print("🔍 Ricerca degli 8 tweet più simili per ciascun tweet di test...")
    sim_matrix = cosine_similarity(test_vecs, train_vecs)
    semantic_contexts = []

    for idx in range(len(df_test)):
        top_k_indices = np.argsort(sim_matrix[idx])[-MAX_K_SHOTS:][::-1]
        similar_examples = [
            {'text': df_train.iloc[i]['text'], 'label': df_train.iloc[i]['label']}
            for i in top_k_indices
        ]
        semantic_contexts.append(format_fewshot_context(similar_examples))

    df_test['fewshot_semantic_minilm_3class_context'] = semantic_contexts

    # --------------------------------------------------------------------------
    # D) SALVATAGGIO CSV FINALE
    # --------------------------------------------------------------------------
    df_test.to_csv(OUTPUT_DATASET_PATH, index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print(f"🎉 DATASET AUMENTATO GENERATO CON SUCCESSO: {OUTPUT_DATASET_PATH}")
    print(f"📊 Totale tweet processati: {len(df_test)}")
    print("=" * 80)
    print("\nColonna aggiunta:")
    print(" - 'fewshot_random_variable_3class_context' (8 esempi casuali dinamici per riga)")

if __name__ == "__main__":
    generate_augmented_dataset()
