#================================================================================
# SCRIPT BATCH INFERENZA MULTI-MODEL & MULTI-K (0-SHOT + FEW-SHOT) SU DEEPINFRA
#================================================================================

import json
import os
import sys
import time
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print('❌ Libreria openai non installata. Esegui: pip install -q openai')
    sys.exit(1)

# ==============================================================================
# 1. SETUP DEEPINFRA & LISTA MODELLI DA TESTARE
# ==============================================================================
try:
    from google.colab import userdata
    deepinfra_api_key = userdata.get('DEEPINFRA_API_KEY')
except Exception:
    deepinfra_api_key = os.environ.get('DEEPINFRA_API_KEY') or os.environ.get('DEEPINFRA_TOKEN')

if not deepinfra_api_key:
    deepinfra_api_key = input('⚠️ Inserisci la tua DEEPINFRA_API_KEY: ')

client = OpenAI(
    base_url='https://api.deepinfra.com/v1/openai',
    api_key=deepinfra_api_key,
)

# Modelli da testare in sequenza
MODELS_TO_TEST = [
    "google/gemma-3-12b-it"
]

# Modalità ed esperimenti
# '0shot' viene trattato separatamente nel ciclo principale
EVAL_MODES = ['0shot', 'random', 'semantic'] 
K_SHOTS_TO_TEST = [3, 5, 8]  

TEST_DATASET_PATH = 'test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8.csv'
CHECKPOINT_DIR = 'sentipolc_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# 2. HELPER FUNZIONI PROMPT E SLICING CONTESTO
# ==============================================================================
def slice_fewshot_context(full_context_str: str, k: int) -> str:
    """Estrae solo i primi K blocchi di esempio dal testo del contesto."""
    if not isinstance(full_context_str, str) or not full_context_str.strip():
        return ""
    blocks = full_context_str.strip().split("\n\n")
    return "\n\n".join(blocks[:k])

def predict_sentiment(tweet_text: str, model_name: str, fewshot_context: str = None) -> tuple[str, float]:
    """Genera il prompt (0-shot o few-shot) ed effettua la chiamata API."""
    if fewshot_context:
        prompt = (
            "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
            "Di seguito trovi alcuni esempi di riferimento per orientarti:\n\n"
            f"{fewshot_context}\n\n"
            "Ora analizza questo tweet:\n"
            f'Testo: "{tweet_text}"\n'
            "Rispondi SOLTANTO con una parola (Positivo, Negativo, Neutro).\n"
            "Sentiment:"
        )
    else:
        # Prompt dedicato alla modalità 0-shot
        prompt = (
            "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
            f'Testo: "{tweet_text}"\n'
            "Rispondi SOLTANTO con una parola (Positivo, Negativo, Neutro).\n"
            "Sentiment:"
        )

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            latency = time.time() - t0
            raw_response = response.choices[0].message.content.strip()
            return raw_response, latency

        except Exception as e:
            wait_time = 3.0 * (2 ** (attempt - 1))
            print(f"\n⚠️ [ERRORE API] (Tentativo {attempt}/5). Attendo {wait_time}s... Errore: {e}")
            time.sleep(wait_time)

    return "ERRORE", time.time() - t0

def clean_and_map_prediction(raw_pred: str) -> tuple[int, int]:
    clean = raw_pred.lower().strip()
    if 'positivo' in clean:
        return 1, 0
    elif 'negativo' in clean:
        return 0, 1
    elif 'neutro' in clean:
        return 0, 0
    else:
        return 0, 0

# ==============================================================================
# 3. CARICAMENTO DATASET
# ==============================================================================
if not os.path.exists(TEST_DATASET_PATH):
    print(f"❌ File dataset non trovato: {TEST_DATASET_PATH}. Verificare il percorso.")
    sys.exit(1)

df_test = pd.read_csv(TEST_DATASET_PATH)
total_records = len(df_test)
print(f'📊 Dataset Caricato: {total_records} tweet da {TEST_DATASET_PATH}')

# ==============================================================================
# 4. CICLO PRINCIPALE DI TEST MULTI-MODEL, MULTI-MODE E MULTI-K
# ==============================================================================
for selected_model in MODELS_TO_TEST:
    model_slug = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')
    
    # Costruzione griglia di esperimenti per il modello corrente
    experiments = []
    for mode in EVAL_MODES:
        if mode == '0shot':
            experiments.append(('0shot', '0shot', None))
        else:
            for k in K_SHOTS_TO_TEST:
                experiments.append((f'{mode}-{k}shot', mode, k))

    for protocol_mode, eval_mode, k_shots in experiments:
        
        # Nome file checkpoint differenziato per modello e modalità
        if eval_mode == '0shot':
            checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-fmt2-0shot-{model_slug}.json')
            fewshot_col = None
        else:
            checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-fmt2-{eval_mode}-{k_shots}shot-{model_slug}.json')
            fewshot_col = 'fewshot_random_3class_context' if eval_mode == 'random' else 'fewshot_semantic_minilm_3class_context'

        print("\n" + "=" * 80)
        print(f"🚀 AVVIO ESPERIMENTO: [{protocol_mode.upper()}] su [{selected_model}]")
        print(f"📁 Checkpoint destinazione: {checkpoint_file}")
        print("=" * 80)

        results = []
        processed_ids = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data.get("results", [])
                    processed_ids = {str(r["id"]) for r in results}
                    print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} tweet già completati.")
            except Exception as e:
                print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero per questo test.")

        if len(results) == total_records:
            print(f"✅ Esperimento [{protocol_mode.upper()}] già COMPLETATO al 100%. Salto al successivo.")
            continue

        fmt2_data = {
            "metadata": {
                "model": {"provider": "deepinfra", "model_id": selected_model},
                "protocol": {"mode": protocol_mode},
                "dataset": {"total_records": total_records}
            },
            "results": results
        }

        start_time = time.time()

        try:
            for idx, row in df_test.iterrows():
                tweet_id = str(row["id"])
                if tweet_id in processed_ids:
                    continue

                tweet_text = str(row["text"])
                
                # Slicing contestuale se pochi-shot, altrimenti None per 0-shot
                if eval_mode == '0shot':
                    k_shot_ctx = None
                else:
                    full_ctx = str(row[fewshot_col])
                    k_shot_ctx = slice_fewshot_context(full_ctx, k_shots)

                raw_prediction, latency = predict_sentiment(tweet_text, selected_model, k_shot_ctx)
                
                if raw_prediction == "ERRORE":
                    status = "error"
                    pred_opos, pred_oneg = 0, 0
                else:
                    status = "ok"
                    pred_opos, pred_oneg = clean_and_map_prediction(raw_prediction)

                record = {
                    "id": tweet_id,
                    "status": status,
                    "target_opos": int(row["target_opos"]),
                    "target_oneg": int(row["target_oneg"]),
                    "pred_opos": pred_opos,
                    "pred_oneg": pred_oneg,
                    "pred_raw": raw_prediction,
                    "latency_seconds": round(latency, 4)
                }

                results.append(record)
                fmt2_data["results"] = results
                processed_ids.add(tweet_id)

                print(".", end="", flush=True)

                if len(results) % 20 == 0 or len(results) == total_records:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

                time.sleep(0.3)

        except KeyboardInterrupt:
            print(f"\n🛑 Interrotto dall'utente durante [{protocol_mode.upper()}] su [{selected_model}].")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint salvato ({len(results)}/{total_records} salvati). Arresto totale del batch.")
            sys.exit(0)

        print(f"\n🎉 COMPLETATO [{protocol_mode.upper()}] per [{selected_model}] in {time.time() - start_time:.1f}s")

print("\n" + "=" * 80)
print("🏁 TUTTI GLI ESPERIMENTI SU TUTTI I MODELLI SONO STATI COMPLETATI CON SUCCESSO!")
print("=" * 80)
