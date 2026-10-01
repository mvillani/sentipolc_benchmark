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
# 1. CONFIGURAZIONE MODELLO E STRATEGIA
# ==============================================================================
# Scegli il Modello: '1' per Llama 3.1 8B | '2' per Llama 3.3 70B
MODEL_CHOICE = '2'

# Scegli la Strategia: '0-shot' | '3-shot-random' | '3-shot-semantic'
STRATEGY_CHOICE = '0-shot'
#STRATEGY_CHOICE = '3-shot-random'
#STRATEGY_CHOICE = '3-shot-semantic'

MODELS_MAP = {
    '1': 'meta-llama/llama-3.1-8b-instruct',
    '2': 'meta-llama/llama-3.3-70b-instruct',
}

SELECTED_MODEL = MODELS_MAP[MODEL_CHOICE]
EVAL_ENGINE = 'OpenRouter API'
model_slug = (
    SELECTED_MODEL.replace('/', '_').replace('-', '_').replace('.', '_')
)

# Mappatura dei File di Input e Output in base alla strategia
if STRATEGY_CHOICE == '0-shot':
  STRATEGY_TYPE = '0-Shot (Pure Zero-Shot Prompting)'
  INPUT_CSV = 'test_set_sentipolc16_gold2000.csv'
  FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_zeroshot.csv'
  FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_zeroshot.json'
elif STRATEGY_CHOICE == '3-shot-random':
  STRATEGY_TYPE = '3-Shot Random (Fixed Seed 42)'
  INPUT_CSV = 'test_set_with_random_examples_seed42.csv'
  FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_random_fewshot.csv'
  FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_random_fewshot.json'
elif STRATEGY_CHOICE == '3-shot-semantic':
  STRATEGY_TYPE = '3-Shot Semantic (k-NN Sentence-Transformers)'
  INPUT_CSV = (
      'test_set_with_semantic_examples_sentence_transformers_paraphrase_multilingual_MiniLM_L12_v2.csv'
  )
  FINAL_RESULTS_CSV = f'results_sentipolc_{model_slug}_semantic_fewshot.csv'
  FINAL_METRICS_JSON = f'metrics_sentipolc_{model_slug}_semantic_fewshot.json'
else:
  raise ValueError(
      'STRATEGY_CHOICE non valida. Usa: 0-shot, 3-shot-random o 3-shot-semantic'
  )

TEST_DATASET_NAME = 'EVALITA SentiPOLC 2016 Gold Test (2000 tweet)'

# ==============================================================================
# 2. SETUP API KEY SU OPENROUTER
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

# ==============================================================================
# 3. STAMPA HEADER IDENTIFICATIVO E RIEPILOGATIVO DEL TEST
# ==============================================================================
print('=' * 80)
print('📋 RIEPILOGO SETUP ESERCIZIO SPERIMENTALE')
print('=' * 80)
print(f'🤖 Modello selezionato:  {SELECTED_MODEL}')
print(f'🌐 Aggregatore / Engine: {EVAL_ENGINE}')
print(f'🎯 Dataset di Test:      {TEST_DATASET_NAME}')
print(f'🧪 Tipologia Esercizio:  {STRATEGY_TYPE}')
print(f'📁 File Input:          {INPUT_CSV}')
print(f'💾 File Output CSV:      {FINAL_RESULTS_CSV}')
print('=' * 80 + '\n')

# ==============================================================================
# 4. SYSTEM PROMPT BASE & SCHEMA OUTPUT
# ==============================================================================
BASE_SYSTEM_PROMPT = """Sei un annotatore esperto di Sentiment Analysis per il dataset italiano SentiPOLC.
Per ogni tweet DEVI PRIMA spiegare brevemente la motivazione del tono e POI determinare le due polarità indipendenti opos e oneg.

- opos = 1: se sono presenti parole ed espressioni di apprezzamento, supporto, gioia o elogio, altrimenti 0.
- oneg = 1: se sono presenti parole ed espressioni di critica, attacco politico, sarcasmo, ironia o insulto, altrimenti 0.
"""


# ==============================================================================
# 5. INFERENZA CON PARSER UNIVERSALE E EXPONENTIAL BACKOFF
# ==============================================================================
def predict_sentiment_llama(
    text: str, context_examples: str, tweet_id: str
) -> dict:
  if context_examples:
    system_prompt = f'{BASE_SYSTEM_PROMPT}\n{context_examples}'
    user_prompt = (
        f'Tweet: "{text}"\nCompila la motivazione (max 15 parole) e poi i'
        ' valori numerici:'
    )
  else:
    system_prompt = (
        f'{BASE_SYSTEM_PROMPT}\n\nRispondi ESCLUSIVAMENTE con un oggetto JSON'
        ' valido con questa struttura:\n{"motivazione": "breve spiegazione",'
        ' "opos": 0 o 1, "oneg": 0 o 1}'
    )
    user_prompt = (
        f'Tweet da analizzare: "{text}"\nFornisci la motivazione e le due'
        " polarita' opos e oneg in formato JSON:"
    )

  max_attempts = 5
  for attempt in range(1, max_attempts + 1):
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
              'X-Title': 'SentiPOLC Llama Evaluation',
          },
      )

      raw_text = response.choices[0].message.content
      data = json.loads(raw_text)

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
      wait_time = attempt * 2
      print(
          f'\n⚠️ [ATTESA RATE-LIMIT/API] Tweet ID {tweet_id} (Tentativo'
          f' {attempt}/{max_attempts}): {type(e).__name__}. Pausa di'
          f' {wait_time}s...'
      )
      time.sleep(wait_time)

  return {
      'motivazione': 'FALLIMENTO_API_DOPO_5_TENTATIVI',
      'opos': 0,
      'oneg': 0,
      'raw_prompt_user': user_prompt,
      'raw_system_prompt': system_prompt,
      'raw_response': 'ERROR',
  }


# ==============================================================================
# 6. FUNZIONE CARICAMENTO DATASET
# ==============================================================================
def load_dataset(filepath: str) -> pd.DataFrame:
  if 'fewshot_prompt_context' in pd.read_csv(filepath, nrows=1).columns:
    return pd.read_csv(filepath)

  # Parsing specifico per il dataset grezzo SentiPOLC
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
              'fewshot_prompt_context': '',
          })
  return pd.DataFrame(rows)


# ==============================================================================
# 7. CARICAMENTO DATASET & CHECKPOINT RESTART
# ==============================================================================
df_test = load_dataset(INPUT_CSV)
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
last_batch_sample = None

print(
    f'🚀 Avvio/Ripresa analisi {STRATEGY_TYPE} per [{SELECTED_MODEL}] su'
    f' {len(df_test)} tweet...\n'
)

for idx, row in df_test.iterrows():
  tweet_id = str(row['id'])

  if tweet_id in processed_ids:
    continue

  tweet_text = str(row['text'])
  context_examples = str(row.get('fewshot_prompt_context', ''))
  if pd.isna(context_examples) or context_examples == 'nan':
    context_examples = ''

  pred = predict_sentiment_llama(tweet_text, context_examples, tweet_id)

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
  last_batch_sample = res_item

  print('.', end='', flush=True)

  if (len(results)) % BATCH_SIZE == 0 or (idx + 1) == len(df_test):
    current_count = len(results)
    elapsed = time.time() - start_time
    speed = current_count / elapsed if elapsed > 0 else 0

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
# 8. SALVATAGGIO DEFINITIVO E METRICHE FINALI
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
