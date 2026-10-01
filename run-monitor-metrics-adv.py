import json
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

# ==============================================================================
# 1. CONFIGURAZIONE PERCORSI E FILTRI
# ==============================================================================
CHECKPOINT_DIR = "sentipolc_eval"  # O "/content/drive/MyDrive/sentipolc_eval" su Colab
FILE_PREFIX = "checkpoint-fmt2-"

def load_fmt2_checkpoint(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    Carica un file di checkpoint fmt2 e converte i risultati in un DataFrame.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    meta = data.get("metadata", {})
    results = [r for r in data.get("results", []) if r.get("status") == "ok"]
    
    df = pd.DataFrame(results)
    return meta, df

def map_to_3class(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Mappa le coppie opos/oneg nelle 3 classi tradizionali: Positivo, Negativo, Neutro.
    (I casi misti opos=1 e oneg=1 vengono etichettati come Misto/Sarcasmo se presenti).
    """
    def resolve_label(opos, oneg):
        if opos == 1 and oneg == 0:
            return "Positivo"
        elif opos == 0 and oneg == 1:
            return "Negativo"
        elif opos == 0 and oneg == 0:
            return "Neutro"
        else:
            return "Misto"  # Caso opos=1 e oneg=1 (es. SentiPOLC Sarcasmo)

    y_true = [resolve_label(r['target_opos'], r['target_oneg']) for _, r in df.iterrows()]
    y_pred = [resolve_label(r['pred_opos'], r['pred_oneg']) for _, r in df.iterrows()]
    
    return pd.Series(y_true), pd.Series(y_pred)

# ==============================================================================
# 2. MOTORE DI CALCOLO METRICHE & MATRICE DI CONFUSIONE
# ==============================================================================
def evaluate_fmt2_file(filepath: str):
    meta, df = load_fmt2_checkpoint(filepath)
    
    if df.empty:
        print(f"⚠️ File vuoto o senza record 'ok': {filepath}")
        return

    model_id = meta.get("model", {}).get("model_id", "Unknown Model")
    provider = meta.get("model", {}).get("provider", "Unknown Provider")
    mode = meta.get("protocol", {}).get("mode", "Unknown Mode")
    
    print("\n" + "=" * 80)
    print(f"📊 REPORT METRICHE: {model_id} ({provider.upper()}) | Modalità: {mode}")
    print(f"📂 File: {os.path.basename(filepath)} | Record analizzati: {len(df)}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # A) METRICHE STANDARD SENTIPOLC (Binary POS / NEG)
    # --------------------------------------------------------------------------
    acc_pos = accuracy_score(df['target_opos'], df['pred_opos']) * 100
    acc_neg = accuracy_score(df['target_oneg'], df['pred_oneg']) * 100
    f1_pos = f1_score(df['target_opos'], df['pred_opos'], average='macro')
    f1_neg = f1_score(df['target_oneg'], df['pred_oneg'], average='macro')
    combined_f1 = (f1_pos + f1_neg) / 2.0

    print("\n🔹 [METRICHE STANDARD SENTIPOLC]")
    print(f"   - Accuracy OPOS: {acc_pos:.2f}% | F1-Macro OPOS: {f1_pos:.4f}")
    print(f"   - Accuracy ONEG: {acc_neg:.2f}% | F1-Macro ONEG: {f1_neg:.4f}")
    print(f"   👉 COMBINED F1-SCORE: {combined_f1:.4f}")

    # --------------------------------------------------------------------------
    # B) ANALISI DEL NEUTRO E METRICHE A 3 CLASSI
    # --------------------------------------------------------------------------
    y_true_3c, y_pred_3c = map_to_3class(df)
    
    # Maschera specifica per la classe Neutro
    is_true_neutro = (y_true_3c == "Neutro")
    is_pred_neutro = (y_pred_3c == "Neutro")
    
    acc_neutro = accuracy_score(is_true_neutro, is_pred_neutro) * 100
    f1_neutro = f1_score(is_true_neutro, is_pred_neutro, average='binary')

    print("\n🔹 [ANALISI SPECIFICA CLASSE NEUTRO]")
    print(f"   - Accuracy Neutro (Binary Is-Neutro): {acc_neutro:.2f}%")
    print(f"   - F1-Score Neutro:                    {f1_neutro:.4f}")
    print(f"   - Totale Tweet Neutri Reali:           {is_true_neutro.sum()} / {len(df)}")
    print(f"   - Totale Tweet Predetti Neutri:        {is_pred_neutro.sum()} / {len(df)}")

    # --------------------------------------------------------------------------
    # C) MATRICE DI CONFUSIONE 3X3
    # --------------------------------------------------------------------------
    labels = ["Positivo", "Negativo", "Neutro"]
    
    # Filtra eventuali classi "Misto" per mantenere la matrice pulita 3x3
    valid_mask = y_true_3c.isin(labels) & y_pred_3c.isin(labels)
    cm = confusion_matrix(y_true_3c[valid_mask], y_pred_3c[valid_mask], labels=labels)
    
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Reale {l}" for l in labels], 
        columns=[f"Pred {l}" for l in labels]
    )

    print("\n🔹 [MATRICE DI CONFUSIONE 3x3]")
    print(cm_df.to_string())
    
    print("\n🔹 [REPORT DI CLASSIFICAZIONE COMPLETO]")
    print(classification_report(y_true_3c[valid_mask], y_pred_3c[valid_mask], target_names=labels, digits=4))
    
    # --------------------------------------------------------------------------
    # D) DIDASCALIE ESPLICATIVE DELLE METRICHE
    # --------------------------------------------------------------------------
    print("-" * 80)
    print("📌 Legenda delle metriche del report:")
    print("   • Precision: Quanti dei tweet classificati in una classe appartengono davvero ad essa (affidabilità della predizione).")
    print("   • Recall:    Quanti tweet reali di una specifica classe sono stati identificati correttamente (capacità di cattura).")
    print("   • F1-score:  Media armonica tra Precision e Recall; misura l'equilibrio complessivo della classificazione.")
    print("   • Support:   Numero totale di esempi reali presenti nel dataset per ciascuna classe.")
    print("=" * 80)

# ==============================================================================
# 3. ESECUZIONE SU TUTTI I FILE FMT2 TROVATI
# ==============================================================================
if __name__ == "__main__":
    search_path = os.path.join(CHECKPOINT_DIR, f"{FILE_PREFIX}*.json")
    fmt2_files = glob.glob(search_path)
    
    if not fmt2_files:
        print(f"❌ Nessun file checkpoint trovato in '{search_path}'. Verificare il percorso.")
    else:
        print(f"🔍 Trovati {len(fmt2_files)} file di checkpoint. Avvio analisi...")
        for filepath in sorted(fmt2_files):
            try:
                evaluate_fmt2_file(filepath)
            except Exception as e:
                print(f"❌ Errore durante l'elaborazione di {filepath}: {e}")
