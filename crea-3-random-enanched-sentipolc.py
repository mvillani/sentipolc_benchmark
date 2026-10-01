# =================================================================================================================
# Script per generare il file con i 3 esempi Random (Una tantum)
# script una tantum . seed casuale (random_state=42) per garantire la riproducibilità,
# estrarrà 3 tweet dal dataset di training e creerà il file test_set_with_random_examples_seed42.csv.
# =================================================================================================================
# Log dell'esecuzione:
# Dataset Caricati -> Train: 7410 tweet | Test: 2000 tweet
# =================================================================================================================
# FILE RANDOM FEW-SHOT SALVATO: test_set_with_random_examples_seed42.csv
# Tutti i 2.000 tweet di test contengono ora 3 esempi casuali estratti dal Training Set.
# =================================================================================================================

import pandas as pd

TRAIN_DATASET_PATH = 'training_set_sentipolc16.csv'
TEST_DATASET_PATH = 'test_set_sentipolc16_gold2000.csv'
OUTPUT_RANDOM_ENRICHED_TEST = 'test_set_with_random_examples_seed42.csv'
RANDOM_SEED_BASE = 42

# 1. Caricamento Dataset SentiPOLC
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
                        'text': text
                    })
    return pd.DataFrame(rows)

df_train = load_sentipolc_csv(TRAIN_DATASET_PATH)
df_test = load_sentipolc_csv(TEST_DATASET_PATH)

print(f"📊 Dataset Caricati -> Train: {len(df_train)} tweet | Test: {len(df_test)} tweet")

# 2. Estrazione casuale indipendente dei 3 vicini random per ciascun tweet di test
random_contexts = []

for idx in range(len(df_test)):
    # Seed univoco legato all'indice per garantire la riproducibilità esatta
    seed = RANDOM_SEED_BASE + idx
    random_samples = df_train.sample(n=3, random_state=seed)
    
    examples_str = "ESEMPI GUIDA ESTRATTI CASUALMENTE DAL TRAINING SET:\n"
    for i, (_, ex_row) in enumerate(random_samples.iterrows(), 1):
        examples_str += f'{i}. Tweet: "{ex_row["text"]}" -> opos: {ex_row["opos"]}, oneg: {ex_row["oneg"]}\n'
    
    random_contexts.append(examples_str)

# 3. Assegnazione della colonna contesto e salvataggio CSV
df_test['fewshot_prompt_context'] = random_contexts
df_test.to_csv(OUTPUT_RANDOM_ENRICHED_TEST, index=False, encoding='utf-8')

print("\n" + "=" * 70)
print(f"✅ FILE RANDOM FEW-SHOT SALVATO: {OUTPUT_RANDOM_ENRICHED_TEST}")
print("Tutti i 2.000 tweet di test contengono ora 3 esempi casuali estratti dal Training Set.")
print("=" * 70)

