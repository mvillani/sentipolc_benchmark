#versione migliorata nelle stampe
import json
import os
import sys
import time

BATCH_SIZE = 64

# from google.colab import userdata
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
  from openai import OpenAI
except ImportError:
  print('❌ Libreria openai non installata. Esegui: !pip install -q openai')
  sys.exit(1)

# ==============================================================================
# 1. SETUP API KEY & SELEZIONE MODELLO SU OPENROUTER
# ==============================================================================
try:
  openrouter_api_key = userdata.get('OPENROUTER_API_KEY')
except Exception:
  openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')

if not openrouter_api_key:
  openrouter_api_key = input('⚠️ Inserisci la tua OPENROUTER_API_KEY: ')

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=openrouter_api_key,
)

# --- CONFIGURAZIONE METADATI TEST ---
SELECTED_MODEL = (
    'mistralai/mistral-small-24b-instruct-2501'  # 1. Mistral Small 24B
)
# SELECTED_MODEL = "meta-llama/llama-3.1-8b-instruct"          # 2. Llama 3.1 8B
# SELECTED_MODEL = "meta-llama/llama-3.3-70b-instruct"         # 3. Llama 3.3 70B

EVAL_ENGINE = 'OpenRouter API'
STRATEGY_TYPE = '3-Shot Random (Fixed Seed 42)'
TEST_DATASET_NAME = 'EVALITA SentiPOLC 2016 Gold Test (2000 tweet)'

model_slug = (
    SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
)
RANDOM_TEST_CSV = 'test_set_with_random_examples_seed42.csv'

FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_random_fewshot.csv'
FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_random_fewshot.json'

# ==============================================================================
# 2. STAMPA HEADER IDENTIFICATIVO E RIEPILOGATIVO DEL TEST
# ==============================================================================
print('=' * 80)
print('📋 RIEPILOGO SETUP ESERCIZIO SPERIMENTALE')
print('=' * 80)
print(f'🤖 Modello selezionato:  {SELECTED_MODEL}')
print(f'🌐 Aggregatore / Engine: {EVAL_ENGINE}')
print(f'🎯 Dataset di Test:      {TEST_DATASET_NAME}')
print(f'🧪 Tipologia Esercizio:  {STRATEGY_TYPE}')
print(f'📁 File Input:          {RANDOM_TEST_CSV}')
print(f'💾 File Output CSV:      {FINAL_RESULTS_CSV}')
print('=' * 80 + '\n')

# ==============================================================================
# 3. SYSTEM PROMPT BASE & SCHEMA OUTPUT
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto.
"""


# ==============================================================================
# 4. INFERENZA CON PARSER UNIVERSALE ROBUSATO (E TRACCIAMENTO RAW)
# ==============================================================================
def predict_sentiment_openrouter_random(
    text: str, random_examples: str, tweet_id: str
) -> dict:
  system_prompt = f'{BASE_SYSTEM_PROMPT}\n{random_examples}'
  user_prompt = (
      f'Tweet: "{text}"\nCompila la motivazione (max 15 parole) e poi i valori'
      ' numerici:'
  )

  for attempt in range(1, 4):
    try:
      response = client.chat.completions.create(
          model=SELECTED_MODEL,
          messages=[
              {'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': user_prompt},
          ],
          temperature=0.0,
          max_tokens=150,
          response_format={'type': 'json_object'},
          extra_headers={
              'HTTP-Referer': 'https://colab.research.google.com',
              'X-Title': 'SentiPOLC Random Few-Shot Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

      # Parser universale per liste/oggetti annidati
      parsed_obj = data
      if isinstance(parsed_obj, list) and len(parsed_obj) > 0:
        parsed_obj = parsed_obj[0]

      if isinstance(parsed_obj, dict) and not (
          'opos' in parsed_obj or 'oneg' in parsed_obj
      ):
        for key, val in parsed_obj.items():
          if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            parsed_obj = val[0]
            break
          elif isinstance(val, dict) and (
              'opos' in val or 'oneg' in val
          ):
            parsed_obj = val
            break

      if isinstance(parsed_obj, dict):
        opos_val = parsed_obj.get('opos', 0)
        oneg_val = parsed_obj.get('oneg', 0)

        opos = 1 if str(opos_val).lower() in ['1', 'true'] else 0
        oneg = 1 if str(oneg_val).lower() in ['1', 'true'] else 0

        return {
            'motivazione': str(parsed_obj.get('motivazione', ''))[:200],
            'opos': opos,
            'oneg': oneg,
            'raw_prompt_user': user_prompt,
            'raw_system_prompt': system_prompt,
            'raw_response': raw_text,
        }

    except Exception as e:
      print(
          f'\n❌ [ERRORE OPENROUTER] Tweet ID {tweet_id} (Tentativo'
          f' {attempt}/3): {type(e).__name__} - {e}'
      )
      time.sleep(1.0)

  return {
      'motivazione': 'FALLIMENTO_API_DOPO_3_TENTATIVI',
      'opos': 0,
      'oneg': 0,
      'raw_prompt_user': user_prompt,
      'raw_system_prompt': system_prompt,
      'raw_response': 'ERROR',
  }


# ==============================================================================
# 5. CARICAMENTO DATASET & ESECUZIONE CON ISPEZIONE PROMPT PER LOTTO
# ==============================================================================
df_test = pd.read_csv(RANDOM_TEST_CSV)
results = []
processed_ids = set()

if os.path.exists(FINAL_RESULTS_CSV):
  try:
    existing_df = pd.read_csv(FINAL_RESULTS_CSV)
    results = existing_df.to_dict('records')
    processed_ids = set(existing_df['id'].astype(str).tolist())
    print(
        f'🔄 CHECKPOINT TROVATO! Ripresa dall ID: {len(processed_ids)} /'
        f' {len(df_test)} tweet già completati.\n'
    )
  except Exception as e:
    print(f'⚠️ Impossibile leggere il file di checkpoint precedente: {e}')

start_time = time.time()
last_batch_sample = None  # Memorizza un esempio del lotto da mostrare

print(
    f'🚀 Avvio/Ripresa analisi su {len(df_test)} tweet per [{SELECTED_MODEL}]...\n'
)

for idx, row in df_test.iterrows():
  tweet_id = str(row['id'])

  if tweet_id in processed_ids:
    continue

  tweet_text = str(row['text'])
  random_context = str(row['fewshot_prompt_context'])

  pred = predict_sentiment_openrouter_random(
      tweet_text, random_context, tweet_id
  )

  res_item = {
      'id': tweet_id,
      'text': tweet_text,
      'target_opos': int(row['opos']),
      'target_oneg': int(row['oneg']),
      'pred_opos': pred['opos'],
      'pred_oneg': pred['oneg'],
      'motivazione': pred.get('motivazione', ''),
      'raw_system_prompt': pred.get('raw_system_prompt', ''),
      'raw_response': pred.get('raw_response', ''),
  }

  results.append(res_item)
  last_batch_sample = res_item  # Salva il tweet per la stampa del lotto

  print('.', end='', flush=True)

  # Ogni BATCH_SIZE tweet (o alla fine), calcola le metriche e mostra l'ISPEZIONE PROMPT
  if (len(results)) % BATCH_SIZE == 0 or (idx + 1) == len(df_test):
    current_count = len(results)
    elapsed = time.time() - start_time
    speed = current_count / elapsed if elapsed > 0 else 0

    # Salvataggio immediato del checkpoint su file CSV
    pd.DataFrame(results).to_csv(
        FINAL_RESULTS_CSV, index=False, encoding='utf-8'
    )

    y_true_pos = [item['target_opos'] for item in results]
    y_pred_pos = [item['pred_opos'] for item in results]
    y_true_neg = [item['target_oneg'] for item in results]
    y_pred_neg = [item['pred_oneg'] for item in results]

    acc_pos = accuracy_score(y_true_pos, y_pred_pos) * 100
    acc_neg = accuracy_score(y_true_neg, y_pred_neg) * 100
    f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')
    f1_neg = f1_score(y_true_neg, y_pred_neg, average='macro')
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print('\n' + '=' * 80)
    print(
        f'📦 PARZIALE {STRATEGY_TYPE} ({current_count} / {len(df_test)} tweet)'
        f' | Velocità: {speed:.2f} tweet/s'
    )
    print(f'🔹 Accuracy opos: {acc_pos:.2f}% | Accuracy oneg: {acc_neg:.2f}%')
    print(f'-> F1-Macro opos: {f1_pos:.4f} | F1-Macro oneg: {f1_neg:.4f}')
    print(f'👉 COMBINED F1-SCORE PARZIALE: {combined_f1:.4f}')
    print('-' * 80)
    print('🔎 ISPEZIONE ESEMPIO PROMPT/OUTPUT DAL LOTTO:')
    if last_batch_sample:
      print(f"🔸 Tweet Analizzato: \"{last_batch_sample['text']}\"")
      print(
          '🔸 Output Generato dal Modello:'
          f" {last_batch_sample['raw_response']}"
      )
      print(
          f"🔸 Targets Reali: [opos={last_batch_sample['target_opos']},"
          f" oneg={last_batch_sample['target_oneg']}] | Predetti:"
          f" [opos={last_batch_sample['pred_opos']},"
          f" oneg={last_batch_sample['pred_oneg']}]"
      )
    print('=' * 80 + '\n')

# ==============================================================================
# 6. SALVATAGGIO DEFINITIVO E METRICHE FINALI
# ==============================================================================
y_true_pos = [item['target_opos'] for item in results]
y_pred_pos = [item['pred_opos'] for item in results]
y_true_neg = [item['target_oneg'] for item in results]
y_pred_neg = [item['pred_oneg'] for item in results]

f1_pos = f1_score(y_true_pos, y_pred_pos, average='macro')
f1_neg = f1_score(y_true_neg, y_pred_neg, average='macro')
combined_f1 = (f1_pos + f1_neg) / 2.0

final_metrics = {
    'model_name': SELECTED_MODEL,
    'eval_engine': EVAL_ENGINE,
    'strategy': STRATEGY_TYPE,
    'dataset': TEST_DATASET_NAME,
    'acc_opos': accuracy_score(y_true_pos, y_pred_pos) * 100,
    'acc_oneg': accuracy_score(y_true_neg, y_pred_neg) * 100,
    'f1_opos': f1_pos,
    'f1_oneg': f1_neg,
    'combined_f1': combined_f1,
    'total_tweets': len(results),
    'elapsed_time_sec': time.time() - start_time,
}

pd.DataFrame(results).to_csv(FINAL_RESULTS_CSV, index=False, encoding='utf-8')
with open(FINAL_METRICS_JSON, 'w', encoding='utf-8') as f:
  json.dump(final_metrics, f, ensure_ascii=False, indent=2)

print('\n' + '=' * 80)
print(f'🎉 VALUTAZIONE {STRATEGY_TYPE} COMPLETATA PER {SELECTED_MODEL}!')
print(f"⏱️ Tempo totale: {final_metrics['elapsed_time_sec']:.1f} secondi")
print(f"🔹 Accuracy opos: {final_metrics['acc_opos']:.2f}%")
print(f"🔹 Accuracy oneg: {final_metrics['acc_oneg']:.2f}%")
print(f"-> F1-Macro opos: {final_metrics['f1_opos']:.4f}")
print(f"-> F1-Macro oneg: {final_metrics['f1_oneg']:.4f}")
print(f"👉 COMBINED F1-SCORE DEFINITIVO: {final_metrics['combined_f1']:.4f}")
print('=' * 80)
