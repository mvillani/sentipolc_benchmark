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
# 1. SETUP DEEPINFRA & SELEZIONE MODELLO
# ==============================================================================
try:
    from google.colab import userdata
    deepinfra_api_key = userdata.get('DEEPINFRA_API_KEY')
except Exception:
    deepinfra_api_key = os.environ.get('DEEPINFRA_API_KEY') or os.environ.get('DEEPINFRA_TOKEN')

if not deepinfra_api_key:
    deepinfra_api_key = input('⚠️ Inserisci la tua DEEPINFRA_API_KEY: ')

# Client OpenAI configurato con l'endpoint di DeepInfra
client = OpenAI(
    base_url='https://api.deepinfra.com/v1/openai',
    api_key=deepinfra_api_key,
)

# SELEZIONE MODELLO SU DEEPINFRA
SELECTED_MODEL = 'mistralai/mistral-small-24b-instruct-2501'
# SELECTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# SELECTED_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

TEST_DATASET_PATH = 'test_set_sentipolc16_gold2000.csv'
CHECKPOINT_DIR = 'sentipolc_eval'  # Cartella locale per i checkpoint

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model_slug = SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
CHECKPOINT_FMT2_FILE = os.path.join(CHECKPOINT_DIR, f'checkpoint-fmt2-0shot-{model_slug}.json')

# ==============================================================================
# 2. INFERENZA 0-SHOT (CON DEEPINFRA)
# ==============================================================================
def predict_sentiment_0shot(text: str, tweet_id: str) -> tuple[str, float]:
    """
    Invia un prompt 0-shot direttamente a DeepInfra.
    Ritorna una tupla: (risposta_raw, latenza_in_secondi)
    """
    prompt = (
        "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
        "Rispondi SOLTANTO con una parola.\n\n"
        f'Testo: "{text}"\n'
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
            print(f"\n⚠️ [ERRORE DEEPINFRA] Tweet ID {tweet_id} (Tentativo {attempt}/5). Attendo {wait_time}s... Errore: {e}")
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
# 3. CARICAMENTO DATASET & MAPPATURA TARGET
# ==============================================================================
def load_sentipolc(filepath: str) -> pd.DataFrame:
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
                        'target_opos': int(opos),
                        'target_oneg': int(oneg),
                        'text': text,
                    })
    return pd.DataFrame(rows)

df_test = load_sentipolc(TEST_DATASET_PATH)
total_records = len(df_test)
print(f'📊 Dataset Caricato: {total_records} tweet')

# ==============================================================================
# 4. RIPRESA CHECKPOINT ESISTENTE (SE PRESENTE)
# ==============================================================================
results = []
processed_ids = set()

if os.path.exists(CHECKPOINT_FMT2_FILE):
    try:
        with open(CHECKPOINT_FMT2_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            results = existing_data.get("results", [])
            processed_ids = {r["id"] for r in results}
            print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} tweet gia completati.")
    except Exception as e:
        print(f"⚠️ Errore nel caricamento del checkpoint: {e}. Si riparte da zero.")

# Struttura FMT2 aggiornata con provider "deepinfra"
fmt2_data = {
    "metadata": {
        "model": {
            "provider": "deepinfra",
            "model_id": SELECTED_MODEL
        },
        "protocol": {
            "mode": "0-shot"
        },
        "dataset": {
            "total_records": total_records
        }
    },
    "results": results
}

# ==============================================================================
# 5. ESECUZIONE CON SALVATAGGIO FMT2 E GESTIONE PAUSA/INTERRUZIONE
# ==============================================================================
start_time = time.time()
print(f'🚀 Avvio Baseline 0-Shot FMT2 su DeepInfra [{SELECTED_MODEL}]...\n')

try:
    for idx, row in df_test.iterrows():
        tweet_id = str(row["id"])
        if tweet_id in processed_ids:
            continue

        tweet_text = str(row["text"])
        raw_prediction, latency = predict_sentiment_0shot(tweet_text, tweet_id)
        
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

        # Salvataggio ogni N tweet
        if len(results) % 64 == 0 or len(results) == total_records:
            with open(CHECKPOINT_FMT2_FILE, 'w', encoding='utf-8') as f:
                json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
                
            elapsed = time.time() - start_time
            print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} tweet | File: {CHECKPOINT_FMT2_FILE}")

        # Micro-pausa preventiva tra le chiamate (0.3s)
        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n\n🛑 Esecuzione interrotta dall'utente.")
    with open(CHECKPOINT_FMT2_FILE, 'w', encoding='utf-8') as f:
        json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Checkpoint di emergenza salvato correttamente ({len(results)}/{total_records} tweet).")

else:
    print('\n' + '=' * 60)
    print(f'🎉 ESTRAZIONE COMPLETATA SU DEEPINFRA PER {SELECTED_MODEL}!')
    print(f"⏱️ Tempo totale: {time.time() - start_time:.1f}s")
    print(f"💾 File salvato: {CHECKPOINT_FMT2_FILE}")
    print('=' * 60)
