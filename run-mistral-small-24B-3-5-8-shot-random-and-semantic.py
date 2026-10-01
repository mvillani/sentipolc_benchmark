
#================================================================================
#SCRIPT BATCH INFERENZA MULTI-K FEW-SHOT (3-CLASS) SU DEEPINFRA
#================================================================================
#Esegue automaticamente una griglia di test variando:
#- Modalita: 'random' e 'semantic'
#- Numero di Shot (K): definiti nell'array K_SHOTS_TO_TEST (default: [3, 8])
#
#Tutti i checkpoint vengono salvati nel formato JSON 'fmt2' e sono immediatamente
#leggibili dalla dashboard e dallo script delle metriche riassuntive.
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
# 1. SETUP DEEPINFRA & SELEZIONE BATCH
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

SELECTED_MODEL = 'mistralai/mistral-small-24b-instruct-2501'
# SELECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- GRIGLIA DI TEST AUTOMATICA ---
EVAL_MODES = ['random', 'semantic']  # Modalita da testare
K_SHOTS_TO_TEST = [3, 5, 8]             # Lunghezze di contesto da testare (5 rimosso)

TEST_DATASET_PATH = 'test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8.csv'
CHECKPOINT_DIR = 'sentipolc_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# 2. HELPER FUNZIONI PROMPT E SLICING CONTESTO
# ==============================================================================
def slice_fewshot_context(full_context_str: str, k: int) -> str:
    """Estrae solo i primi K blocchi di esempio dal testo del contesto."""
    blocks = full_context_str.strip().split("\n\n")
    return "\n\n".join(blocks[:k])

def predict_sentiment_fewshot(tweet_text: str, fewshot_context: str, tweet_id: str) -> tuple[str, float]:
    prompt = (
        "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
        "Di seguito trovi alcuni esempi di riferimento per orientarti:\n\n"
        f"{fewshot_context}\n\n"
        "Ora analizza questo tweet:\n"
        f'Testo: "{tweet_text}"\n'
        "Rispondi SOLTANTO con una parola (Positivo, Negativo, Neutro).\n"
        "Sentiment:"
    )

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=SELECTED_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            latency = time.time() - t0
            raw_response = response.choices[0].message.content.strip()
            return raw_response, latency

        except Exception as e:
            wait_time = 3.0 * (2 ** (attempt - 1))
            print(f"\n⚠️ [ERRORE API] Tweet ID {tweet_id} (Tentativo {attempt}/5). Attendo {wait_time}s... Errore: {e}")
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
# 3. CARICAMENTO DATASET AUMENTATO
# ==============================================================================
if not os.path.exists(TEST_DATASET_PATH):
    print(f"❌ File dataset non trovato: {TEST_DATASET_PATH}. Verificare il percorso.")
    sys.exit(1)

df_test = pd.read_csv(TEST_DATASET_PATH)
total_records = len(df_test)
print(f'📊 Dataset Caricato: {total_records} tweet da {TEST_DATASET_PATH}')

# ==============================================================================
# 4. CICLO PRINCIPALE DI TEST MULTI-MODE E MULTI-K
# ==============================================================================
model_slug = SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')

for eval_mode in EVAL_MODES:
    fewshot_col = 'fewshot_random_3class_context' if eval_mode == 'random' else 'fewshot_semantic_minilm_3class_context'

    for k_shots in K_SHOTS_TO_TEST:
        protocol_mode = f'fewshot-{eval_mode}-{k_shots}shot'
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-fmt2-{eval_mode}-{k_shots}shot-{model_slug}.json')

        print("\n" + "=" * 80)
        print(f"🚀 AVVIO ESPERIMENTO: [{protocol_mode.upper()}] su [{SELECTED_MODEL}]")
        print(f"📁 Checkpoint destinazione: {checkpoint_file}")
        print("=" * 80)

        # Ripresa da checkpoint esistente se presente
        results = []
        processed_ids = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data.get("results", [])
                    processed_ids = {str(r["id"]) for r in results}
                    print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} tweet gia completati.")
            except Exception as e:
                print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero per questo test.")

        if len(results) == total_records:
            print(f"✅ Esperimento [{protocol_mode.upper()}] gia COMPLETATO al 100%. Salto al successivo.")
            continue

        fmt2_data = {
            "metadata": {
                "model": {"provider": "deepinfra", "model_id": SELECTED_MODEL},
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
                full_ctx = str(row[fewshot_col])
                k_shot_ctx = slice_fewshot_context(full_ctx, k_shots)

                raw_prediction, latency = predict_sentiment_fewshot(tweet_text, k_shot_ctx, tweet_id)
                
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

                # Salvataggio ogni 20 record o fine dataset
                if len(results) % 20 == 0 or len(results) == total_records:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

                time.sleep(0.3)

        except KeyboardInterrupt:
            print(f"\n🛑 Interrotto dall'utente durante [{protocol_mode.upper()}].")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint salvato ({len(results)}/{total_records} salvati). Arresto totale del batch.")
            sys.exit(0)

        print(f"\n🎉 COMPLETATO [{protocol_mode.upper()}] in {time.time() - start_time:.1f}s")

print("\n" + "=" * 80)
print("🏁 TUTTI GLI ESPERIMENTI DEL BATCH SONO STATI COMPLETATI CON SUCCESSO!")
print("=" * 80)
