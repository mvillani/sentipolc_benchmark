
#================================================================================
#SCRIPT DI INFERENZA FEW-SHOT (RANDOM & SEMANTICO) PER SentiPOLC (3-CLASS)
#================================================================================
#Caratteristiche salienti:
#1. Parametrizzato: permette di scegliere con un interruttore tra Few-Shot Random
#   e Few-Shot Semantico (MiniLM).
#2. Lettura Dinamica: estrae il contesto 5-shot nativamente dalle colonne del CSV
#   aumentato (senza ricalcolare embedding al volo).
#3. Piena Compatibilita FMT2: genera file di checkpoint JSON con metadati, latenza 
#   e polarita numeriche per la dashboard ed il report di metriche 3x3.
#4. Resilienza & Ripresa: include pausa preventiva (0.3s), backoff esponenziale 
#   su errori API e salvataggio atomico ogni 20 record (con supporto a Ctrl+C).
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
# 1. SETUP DEEPINFRA & SELEZIONE MODELLO / MODALITÀ
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

# SELEZIONE MODELLO
SELECTED_MODEL = 'mistralai/mistral-small-24b-instruct-2501'
# SELECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- CONFIGURAZIONE MODALITÀ FEW-SHOT ---
# Opzioni disponibili: 'random' oppure 'semantic'
EVAL_MODE = 'semantic'  

TEST_DATASET_PATH = 'test_set_sentipolc16_BENCHMARK_FULL_3CLASS.csv'
CHECKPOINT_DIR = 'sentipolc_eval'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Mappatura colonne ed etichette per il protocollo FMT2
if EVAL_MODE == 'random':
    FEWSHOT_COL = 'fewshot_random_3class_context'
    PROTOCOL_MODE = 'fewshot-random-5shot'
elif EVAL_MODE == 'semantic':
    FEWSHOT_COL = 'fewshot_semantic_minilm_3class_context'
    PROTOCOL_MODE = 'fewshot-semantic-minilm-5shot'
else:
    raise ValueError("Modalita EVAL_MODE non valida! Scegli tra 'random' o 'semantic'.")

model_slug = SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
CHECKPOINT_FMT2_FILE = os.path.join(CHECKPOINT_DIR, f'checkpoint-fmt2-{EVAL_MODE}-{model_slug}.json')

# ==============================================================================
# 2. PROMPT INFERENZA FEW-SHOT
# ==============================================================================
def predict_sentiment_fewshot(tweet_text: str, fewshot_context: str, tweet_id: str) -> tuple[str, float]:
    """
    Invia il contesto pochi-scatti + il tweet di test e restituisce la risposta raw.
    """
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
    """
    Mappa la risposta testuale nelle polarità opos/oneg per la dashboard.
    """
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
    print(f"❌ File dataset non trovato: {TEST_DATASET_PATH}")
    sys.exit(1)

df_test = pd.read_csv(TEST_DATASET_PATH)
total_records = len(df_test)
print(f'📊 Dataset Caricato: {total_records} tweet')
print(f'⚙️ Modalita attiva: [{EVAL_MODE.upper()}] -> Colonna utilizzata: {FEWSHOT_COL}')

# ==============================================================================
# 4. RIPRESA CHECKPOINT ESISTENTE
# ==============================================================================
results = []
processed_ids = set()

if os.path.exists(CHECKPOINT_FMT2_FILE):
    try:
        with open(CHECKPOINT_FMT2_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            results = existing_data.get("results", [])
            processed_ids = {str(r["id"]) for r in results}
            print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} tweet gia completati.")
    except Exception as e:
        print(f"⚠️ Errore caricamento checkpoint: {e}. Si riparte da zero.")

fmt2_data = {
    "metadata": {
        "model": {
            "provider": "deepinfra",
            "model_id": SELECTED_MODEL
        },
        "protocol": {
            "mode": PROTOCOL_MODE
        },
        "dataset": {
            "total_records": total_records
        }
    },
    "results": results
}

# ==============================================================================
# 5. CICLO DI ESECUZIONE
# ==============================================================================
start_time = time.time()
print(f'🚀 Avvio Test Few-Shot [{EVAL_MODE.upper()}] su DeepInfra [{SELECTED_MODEL}]...\n')

try:
    for idx, row in df_test.iterrows():
        tweet_id = str(row["id"])
        if tweet_id in processed_ids:
            continue

        tweet_text = str(row["text"])
        fewshot_ctx = str(row[FEWSHOT_COL])

        raw_prediction, latency = predict_sentiment_fewshot(tweet_text, fewshot_ctx, tweet_id)
        
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

        # Salvataggio ogni 20 tweet
        if len(results) % 20 == 0 or len(results) == total_records:
            with open(CHECKPOINT_FMT2_FILE, 'w', encoding='utf-8') as f:
                json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
                
            elapsed = time.time() - start_time
            print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} tweet | File: {CHECKPOINT_FMT2_FILE}")

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n\n🛑 Esecuzione interrotta dall'utente.")
    with open(CHECKPOINT_FMT2_FILE, 'w', encoding='utf-8') as f:
        json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Checkpoint di emergenza salvato ({len(results)}/{total_records} tweet).")

else:
    print('\n' + '=' * 60)
    print(f'🎉 TEST FEW-SHOT [{EVAL_MODE.upper()}] COMPLETATO PER {SELECTED_MODEL}!')
    print(f"⏱️ Tempo totale: {time.time() - start_time:.1f}s")
    print(f"💾 File salvato: {CHECKPOINT_FMT2_FILE}")
    print('=' * 60)
