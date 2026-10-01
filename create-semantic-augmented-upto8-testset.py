
#================================================================================
#STEP 1: GENERATORE DATASET AUMENTATO FEW-SHOT (8-SHOT BASE)
#================================================================================
#Genera un CSV di test contenente fino a 8 esempi per ciascun tweet (sia Random
#che Semantico con MiniLM) etichettati come Positivo, Negativo, Neutro.
#================================================================================


import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Libreria sentence-transformers non installata. Esegui: pip install -q sentence-transformers")
    sys.exit(1)

TRAIN_DATASET_PATH = 'training_set_sentipolc16.csv'
TEST_DATASET_PATH = 'test_set_sentipolc16_gold2000.csv'
OUTPUT_DATASET_PATH = 'test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8.csv'

MODEL_EMBEDDING_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_K_SHOTS = 8

def map_sentiment_to_3class(opos: int, oneg: int) -> str:
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
    formatted_blocks = []
    for idx, ex in enumerate(examples, 1):
        formatted_blocks.append(f'Esempio {idx}:\nTweet: "{ex["text"]}"\nRisposta: {ex["label"]}')
    return "\n\n".join(formatted_blocks)

def generate_benchmark_dataset():
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(TEST_DATASET_PATH):
        print("❌ ERRORE: Assicurati che i file CSV siano presenti nella cartella corrente.")
        return

    print("📖 Caricamento Training Set...")
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

    # A) FEW-SHOT RANDOM (8 esempi fissi)
    print("\n🎲 Generazione Few-Shot Casuale (8-Shot)...")
    random.seed(42)
    random_indices = random.sample(range(len(df_train)), MAX_K_SHOTS)
    random_examples = [
        {'text': df_train.iloc[idx]['text'], 'label': df_train.iloc[idx]['label']}
        for idx in random_indices
    ]
    df_test['fewshot_random_3class_context'] = format_fewshot_context(random_examples)

    # B) FEW-SHOT SEMANTICO (Top-8 cosine similarity)
    print(f"\n🧠 Caricamento SentenceTransformer [{MODEL_EMBEDDING_NAME}]...")
    embedder = SentenceTransformer(MODEL_EMBEDDING_NAME)

    print("⚡ Calcolo embedding per Training Set e Test Set...")
    train_vecs = embedder.encode(df_train['text'].tolist(), show_progress_bar=False, batch_size=64)
    test_vecs = embedder.encode(df_test['text'].tolist(), show_progress_bar=False, batch_size=64)

    print("🔍 Ricerca degli 8 tweet più simili...")
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

    df_test.to_csv(OUTPUT_DATASET_PATH, index=False, encoding='utf-8')
    print(f"\n🎉 DATASET BENCHMARK 8-SHOT PRONTO: {OUTPUT_DATASET_PATH}")

if __name__ == "__main__":
    generate_benchmark_dataset()
