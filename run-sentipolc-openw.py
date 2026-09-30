#===========================================================================================
# SCRIPT BATCH COLAB MODELLI OPEN WEIGHTS casi mancanti (DEEPINFRA): BENCHMARK 3-CLASS (FMT2 REVISED)
#===========================================================================================
# Esegue la sequenza completa di esperimenti tramite l'API di DeepInfra:
# - 0-shot
# - 0-shot-alt
# - N-shot-random-fixed (K in [3, 5, 8])
# - N-shot-random-variable (K in [3, 5, 8])
# - N-shot-semantic-minilm (K in [3, 5, 8])
#
# Salva i checkpoint FMT2 arricchiti con metadati completi di riproducibilita.
#===========================================================================================

import json
import os
import re
import time
import requests
import pandas as pd
from google.colab import drive, userdata

# ==============================================================================
# 1. CONFIGURAZIONE ESPERIMENTI E AMBIENTE
# ==============================================================================

# API_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# API_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# API_MODEL = "mistralai/mistral-small-24b-instruct-2501"
API_MODEL = "google/gemma-3-12b-it"

CHECKPOINT_EVERY = 64   # Salvataggio atomico su file
STATUS_EVERY = 64       # Frequenza dei messaggi di stato a schermo (quando VERBOSE=False)
VERBOSE = False         # Impostare a True per ripristinare la stampa riga per riga

TEMPERATURE = 0.0
MAX_TOKENS = 10

DRIVE_DIR = "/content/drive/MyDrive/sentipolc_eval"

# Dataset aumentato aggiornato (Versione II)
DATASET_PATH = os.path.join(
    DRIVE_DIR,
    "test_set_sentipolc16_BENCHMARK_FULL_3CLASS_UPTO8_II.csv"
)

# Modalita da eseguire
# EVAL_MODES = ["0-shot", "0-shot-alt", "random-fixed", "random-variable", "semantic-minilm"]
EVAL_MODES = ["random-variable"]
K_SHOTS_TO_TEST = [3, 5, 8]  # Testato su K = 3, 5 e 8 per le varianti few-shot

OVERWRITE = False  # Metti a True solo se vuoi ricominciare gli esperimenti da zero

# ==============================================================================
# 2. INIZIALIZZAZIONE DEEPINFRA & GOOGLE DRIVE
# ==============================================================================
drive.mount("/content/drive")
os.makedirs(DRIVE_DIR, exist_ok=True)

try:
    deepinfra_api_key = userdata.get("DEEPINFRA_API_KEY")
except Exception:
    deepinfra_api_key = os.environ.get("DEEPINFRA_API_KEY")

if not deepinfra_api_key:
    deepinfra_api_key = input("⚠️ Inserisci la tua DEEPINFRA_API_KEY: ")

DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {deepinfra_api_key}",
    "Content-Type": "application/json",
}

# ==============================================================================
# 3. HELPER FUNZIONI: SLICING CONTESTO & MAPPATURA
# ==============================================================================
def slice_fewshot_context(full_context_str: str, k: int) -> str:
    """Estrae solo i primi K blocchi di esempio dal testo del contesto."""
    if not isinstance(full_context_str, str) or not full_context_str.strip():
        return ""
    blocks = full_context_str.strip().split("\n\n")
    return "\n\n".join(blocks[:k])

def clean_and_map_prediction(raw_pred: str) -> tuple[int, int]:
    """Mappa la risposta mono-parola nelle polarita opos/oneg per il formato fmt2."""
    clean = raw_pred.lower().strip()
    if "positivo" in clean:
        return 1, 0
    elif "negativo" in clean:
        return 0, 1
    elif "neutro" in clean:
        return 0, 0
    else:
        return 0, 0

def predict_sentiment_llm(
    text: str, mode: str, k_shots: int = 0, fewshot_ctx: str = ""
) -> tuple[str, float]:
    """Invia il prompt mono-parola al modello tramite DeepInfra API."""
    if mode == "0-shot":
        prompt = (
            "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
            "Rispondi SOLTANTO con una parola.\n\n"
            f'Testo: "{text}"\n'
            "Sentiment:"
        )
    elif mode == "0-shot-alt":
        prompt = (
            "Analizza il testo seguente e indica se il sentiment complessivo e' Positivo, Negativo oppure Neutro.\n"
            "Restituisci unicamente la singola etichetta senza punteggiatura o commenti.\n\n"
            f'Testo: "{text}"\n'
            "Risposta:"
        )
    else:
        prompt = (
            "Determina il sentiment del seguente testo (Positivo, Negativo, Neutro).\n"
            "Di seguito trovi alcuni esempi di riferimento per orientarti:\n\n"
            f"{fewshot_ctx}\n\n"
            "Ora analizza questo tweet:\n"
            f'Testo: "{text}"\n'
            "Rispondi SOLTANTO con una parola (Positivo, Negativo, Neutro).\n"
            "Sentiment:"
        )

    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            response = requests.post(
                url=DEEPINFRA_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=30
            )
            latency = time.time() - t0

            if response.status_code == 200:
                res_json = response.json()
                raw_response = res_json["choices"][0]["message"]["content"].strip()
                return raw_response, latency
            else:
                raise ValueError(f"HTTP Status {response.status_code}: {response.text}")

        except Exception as e:
            wait_time = 2.0 * (2 ** (attempt - 1))
            print(f"\n⚠️ [ERRORE DEEPINFRA] Attempt {attempt}/5. Attesa {wait_time}s... Errore: {e}")
            time.sleep(wait_time)

    return "ERRORE", time.time() - t0

# ==============================================================================
# 4. CARICAMENTO DATASET
# ==============================================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Impossibile trovare il dataset esteso in: {DATASET_PATH}")

df_test = pd.read_csv(DATASET_PATH)
total_tweets = len(df_test)
pad_width = len(str(total_tweets))

print(f"📊 Dataset Caricato: {total_tweets} tweet da {DATASET_PATH}\n")

# ==============================================================================
# 5. CICLO MASTER ESPERIMENTI IN FORMATO FMT2
# ==============================================================================
model_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", API_MODEL).lower()

# Costruzione della lista dei protocolli con la NUOVA NOMENCLATURA
experiments_list = []
for mode in EVAL_MODES:
    if mode in ["0-shot", "0-shot-alt"]:
        experiments_list.append((mode, 0, mode))
    else:
        for k in K_SHOTS_TO_TEST:
            protocol_name = f"{k}-shot-{mode}"
            experiments_list.append((mode, k, protocol_name))

print(f"🚀 AVVIO SEQUENZA ESPERIMENTI ({len(experiments_list)} CONFIGURAZIONI TOTALI SU {API_MODEL})\n")

for exp_index, (eval_mode, k_shots, protocol_mode) in enumerate(experiments_list, 1):
    checkpoint_file = os.path.join(
        DRIVE_DIR, f"checkpoint-fmt2-{protocol_mode}-{model_slug}.json"
    )

    print("=" * 80)
    print(f"▶️ ESPERIMENTO {exp_index}/{len(experiments_list)}: [{protocol_mode.upper()}] su [{API_MODEL}]")
    print(f"📁 Checkpoint FMT2: {checkpoint_file}")
    print("=" * 80)

    if OVERWRITE and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"🗑️ OVERWRITE=True: Reset checkpoint ({checkpoint_file})")

    results = []
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
                processed_ids = {str(r["id"]) for r in results}
            print(f"🔄 RIPRESA AUTOMATICA: {len(processed_ids)}/{total_tweets} tweet gia elaborati.")
        except Exception as e:
            print(f"⚠️ Errore nel caricamento checkpoint ({e}). Si riparte da zero.")

    if len(results) == total_tweets:
        print(f"✅ Esperimento [{protocol_mode.upper()}] gia COMPLETATO al 100%. Salto al successivo.\n")
        continue

    # NUOVA STRUTTURA METADATI ARRICCHITA
    fmt2_data = {
        "metadata": {
            "model": {
                "provider": "deepinfra",
                "model_id": API_MODEL
            },
            "protocol": {
                "mode": protocol_mode
            },
            "experiment": {
                "eval_mode": eval_mode,
                "k_shots": k_shots,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "status_every": STATUS_EVERY
            },
            "dataset": {
                "path": DATASET_PATH,
                "total_records": total_tweets
            }
        },
        "results": results,
    }

    # Selezione dinamica della colonna contestuale
    fewshot_col = ""
    if eval_mode == "random-fixed":
        fewshot_col = "fewshot_random_3class_context"
    elif eval_mode == "random-variable":
        fewshot_col = "fewshot_random_variable_3class_context"
    elif eval_mode == "semantic-minilm":
        fewshot_col = "fewshot_semantic_minilm_3class_context"

    start_time = time.time()

    try:
        for idx, row in df_test.iterrows():
            tweet_id = str(row["id"])
            if tweet_id in processed_ids:
                continue

            tweet_text = str(row["text"])
            target_opos = int(row["target_opos"]) if "target_opos" in row else int(row["opos"])
            target_oneg = int(row["target_oneg"]) if "target_oneg" in row else int(row["oneg"])

            k_shot_ctx = ""
            if eval_mode not in ["0-shot", "0-shot-alt"] and fewshot_col in row:
                full_ctx = str(row[fewshot_col])
                k_shot_ctx = slice_fewshot_context(full_ctx, k_shots)

            raw_pred, latency = predict_sentiment_llm(
                tweet_text, eval_mode, k_shots, k_shot_ctx
            )

            if raw_pred == "ERRORE":
                status = "error"
                pred_opos, pred_oneg = 0, 0
            else:
                status = "ok"
                pred_opos, pred_oneg = clean_and_map_prediction(raw_pred)

            record = {
                "id": tweet_id,
                "status": status,
                "target_opos": target_opos,
                "target_oneg": target_oneg,
                "pred_opos": pred_opos,
                "pred_oneg": pred_oneg,
                "pred_raw": raw_pred,
                "latency_seconds": round(latency, 4),
            }

            results.append(record)
            fmt2_data["results"] = results
            processed_ids.add(tweet_id)

            curr_count = len(results)
            idx_str = f"{curr_count:>{pad_width}}"

            # Gestione Stampa Verbosa / Sintetica
            if VERBOSE:
                print(
                    f"[{idx_str}/{total_tweets}] ID: {tweet_id} | Pred: {raw_pred:<10} | Lat: {latency:.2f}s"
                )
            elif curr_count % STATUS_EVERY == 0 or curr_count == total_tweets:
                elapsed_sec = time.time() - start_time
                avg_lat = elapsed_sec / curr_count if curr_count > 0 else 0
                print(
                    f"⚡ Progresso: [{idx_str}/{total_tweets}] tweet elaborati | Lat. media: {avg_lat:.2f}s/tweet"
                )

            # Salvataggio atomico FMT2 su Drive
            if curr_count % CHECKPOINT_EVERY == 0 or curr_count == total_tweets:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(fmt2_data, f, ensure_ascii=False, indent=2)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n🛑 Interruzione manuale durante [{protocol_mode.upper()}].")
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(fmt2_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Checkpoint di emergenza salvato correttamente ({len(results)}/{total_tweets} tweet).")
        break

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\n✅ COMPLETATO ESPERIMENTO [{protocol_mode.upper()}] in {elapsed_min:.2f} minuti!")
    print(f"📁 Salvato Checkpoint FMT2: {checkpoint_file}\n")

print(f"🎉 TUTTI GLI ESPERIMENTI SU DEEPINFRA PER {API_MODEL} SONO STATI ULTIMATI!")
