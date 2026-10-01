#=========================================================================================================================================
# Dataset Caricati -> Train: 7410 | Test: 2000
# Caricamento modello Sentence-Transformers [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2]...
# Loading weights: 100%|██████████| 199/199 [00:00<00:00, 2529.70it/s]
# Encoding Training Set...
# Batches: 100%|██████████| 116/116 [03:36<00:00,  1.87s/it]
# Encoding Test Set...
# Batches: 100%|██████████| 32/32 [01:04<00:00,  2.01s/it]
# Calcolo Cosine Similarity e associazione 3 vicini semantici per ciascun tweet...
#=========================================================================================================================================
# FILE ARRICCHITO SALVATO: test_set_with_semantic_examples_sentence_transformers_paraphrase_multilingual_MiniLM_L12_v2.csv
# Tutti i 2.000 tweet di test contengono ora i 3 vicini semantici estratti con sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
#=========================================================================================================================================

import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_DATASET_PATH = 'training_set_sentipolc16.csv'
TEST_DATASET_PATH = 'test_set_sentipolc16_gold2000.csv'

# Modello Sentence-Transformers
EMBEDDING_MODEL_NAME = (
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Generazione nome file di output dinamico basato sul modello
transformer_slug = (
    EMBEDDING_MODEL_NAME.replace('/', '_').replace('-', '_').replace('.', '_')
)
OUTPUT_ENRICHED_TEST = f'test_set_with_semantic_examples_{transformer_slug}.csv'


# 1. Caricamento Dataset
def load_sentipolc_csv(filepath: str) -> pd.DataFrame:
  rows = []
  with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split(',', 8)
      if len(parts) >= 9:
        tweet_id = parts[0].strip('"')
        opos = parts[2].strip('"')
        oneg = parts[3].strip('"')
        text = parts[8].strip().strip('"')
        if opos.isdigit() and oneg.isdigit():
          rows.append({
              'id': tweet_id,
              'opos': int(opos),
              'oneg': int(oneg),
              'text': text,
          })
  return pd.DataFrame(rows)


df_train = load_sentipolc_csv(TRAIN_DATASET_PATH)
df_test = load_sentipolc_csv(TEST_DATASET_PATH)
print(f'📊 Dataset Caricati -> Train: {len(df_train)} | Test: {len(df_test)}')

# 2. Generazione Embedding
print(f'🔄 Caricamento modello Sentence-Transformers [{EMBEDDING_MODEL_NAME}]...')
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print('⚡ Encoding Training Set...')
train_embeddings = embedder.encode(
    df_train['text'].tolist(), show_progress_bar=True, batch_size=64
)

print('⚡ Encoding Test Set...')
test_embeddings = embedder.encode(
    df_test['text'].tolist(), show_progress_bar=True, batch_size=64
)

# 3. Vector Search + Alive Sign
print(
    '\n🔍 Calcolo Cosine Similarity e associazione 3 vicini semantici per'
    ' ciascun tweet...'
)
k_contexts = []
start_time = time.time()

for test_idx in range(len(df_test)):
  test_vec = test_embeddings[test_idx].reshape(1, -1)
  similarities = cosine_similarity(test_vec, train_embeddings)[0]

  # Top 3 indici di training con similarity più alta
  top_3_idx = np.argsort(similarities)[-3:][::-1]

  examples_str = 'ESEMPI GUIDA SEMANTICAMENTE SIMILI DAL TRAINING SET:\n'
  for i, idx in enumerate(top_3_idx, 1):
    ex_row = df_train.iloc[idx]
    examples_str += (
        f'{i}. Tweet: "{ex_row["text"]}" -> opos: {ex_row["opos"]}, oneg:'
        f' {ex_row["oneg"]}\n'
    )

  k_contexts.append(examples_str)

  # --- ALIVE SIGN ---
  print('.', end='', flush=True)
  if (test_idx + 1) % 100 == 0 or (test_idx + 1) == len(df_test):
    elapsed = time.time() - start_time
    speed = (test_idx + 1) / elapsed
    print(
        f' 🟢 [{test_idx + 1}/{len(df_test)}] Vicini processati | {speed:.1f}'
        ' tweet/s'
    )

# 4. Salvataggio File Unico con nome dinamico
df_test['fewshot_prompt_context'] = k_contexts
df_test.to_csv(OUTPUT_ENRICHED_TEST, index=False, encoding='utf-8')

print('\n' + '=' * 70)
print(f'✅ FILE ARRICCHITO SALVATO: {OUTPUT_ENRICHED_TEST}')
print(
    'Tutti i 2.000 tweet di test contengono ora i 3 vicini semantici estratti con'
    f' {EMBEDDING_MODEL_NAME}.'
)
print('=' * 70)
