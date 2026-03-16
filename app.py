import re, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Chargement du modèle ---
artifacts = joblib.load('model_artifacts.joblib')
lgb_model = artifacts['lgb_model']
xgb_model = artifacts['xgb_model']
cb_model = artifacts['cb_model']
w_lgb = artifacts['w_lgb']
w_xgb = artifacts['w_xgb']
w_cb = artifacts['w_cb']
tfidf = artifacts['tfidf']
tfidf_cols = artifacts['tfidf_cols']
FEATS = artifacts['features']
global_mean = artifacts['global_mean']
best_alpha = artifacts['best_alpha']
PRICE_MIN = artifacts['price_min']
PRICE_MAX = artifacts['price_max']
top_proche = artifacts['top_proche']
quartier_te = artifacts['quartier_te_map']
proche_te = artifacts['proche_de_te_map']
q_ppm2_map = artifacts['q_ppm2_map']
q_ppm2_global = artifacts['q_ppm2_global']

print(f'✅ Modèle chargé — {len(FEATS)} features')


# --- Fonctions utilitaires (identiques au notebook) ---
def extract_price_millions(text):
    if pd.isna(text): return np.nan
    text = re.sub(r'\b\d{8,}\b', '', str(text))
    matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:مليون|مليار|ملايين|مليوني)', text)
    return max(float(m.replace(',', '.')) for m in matches) if matches else np.nan


def smart_text_price(raw_m, surface):
    if pd.isna(raw_m) or raw_m == 0:
        return 0, 0
    mro = raw_m * 100_000
    mru = raw_m * 1_000_000
    if surface > 0:
        mro_ok = 1_000 <= mro / surface <= 100_000
        mru_ok = 1_000 <= mru / surface <= 100_000
        price = mru if (mru_ok and not mro_ok) else mro
    else:
        price = mro
    price = float(np.clip(price, PRICE_MIN, PRICE_MAX))
    return price, 1


def clean_quartier(name):
    if pd.isna(name): return 'inconnu'
    syns = {'tevragh zeina':'tevragh zeina', 'tevragh-zeina':'tevragh zeina',
            'teyarett':'teyarett', 'dar naim':'dar naim', 'dar-naim':'dar naim',
            'toujounine':'toujounine', 'arafat':'arafat', 'ksar':'ksar',
            'riyadh':'riyadh', 'riyad':'riyadh', 'sebkha':'sebkha'}
    return syns.get(str(name).lower().strip(), str(name).lower().strip())


def engineer_single(data):
    """Applique le feature engineering sur un dict de données et retourne un vecteur."""
    df = pd.DataFrame([data])
    
    # Valeurs par défaut
    for col in ['surface_m2', 'nb_chambres', 'nb_salons', 'nb_sdb']:
        df[col] = pd.to_numeric(df.get(col, pd.Series([0])), errors='coerce').fillna(0)
    for col in ['caracteristiques', 'titre', 'description', 'source', 'date_publication']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')
    if 'quartier' not in df.columns:
        df['quartier'] = 'inconnu'
    
    df['quartier'] = df['quartier'].apply(clean_quartier)
    df['text_all'] = df['titre'].fillna('') + ' ' + df['description'].fillna('')
    
    # Text price
    df['raw_m'] = df['text_all'].apply(extract_price_millions)
    res = df.apply(lambda r: smart_text_price(r['raw_m'], r['surface_m2']), axis=1)
    df['text_price'] = [r[0] for r in res]
    df['has_price'] = [r[1] for r in res]
    
    # Caractéristiques
    carac = df['caracteristiques'].fillna('')
    df['has_tf'] = carac.str.contains('Titre foncier', case=False, na=False).astype(int)
    df['has_garage'] = carac.str.contains('Garage', case=False, na=False).astype(int)
    df['has_piscine'] = carac.str.contains('piscine', case=False, na=False).astype(int)
    df['taille_rue'] = carac.str.extract(r'Taille rue:\s*([\d.]+)', expand=False).astype(float).fillna(0)
    df['nb_balcons'] = carac.str.extract(r'(\d+)\s*balcon', expand=False).astype(float).fillna(0)
    df['nb_carac'] = carac.apply(lambda x: len(x.split('|')) if x else 0)
    df['proche_raw'] = (carac.str.extract(r'Proche de:\s*([^|]+)', expand=False)
                        .fillna('none').str.strip().str[:40])
    
    # Mots-clés texte
    tl = df['text_all'].str.lower()
    df['is_villa'] = tl.str.contains('فيلا|villa', na=False).astype(int)
    df['is_duplex'] = tl.str.contains('دوبلكس|دبلكس|duplex', na=False).astype(int)
    df['is_house'] = tl.str.contains('دار|منزل', na=False).astype(int)
    df['is_land'] = tl.str.contains('أرض|ارض|terrain', na=False).astype(int)
    df['is_corner'] = tl.str.contains('ركن|الركن', na=False).astype(int)
    df['is_new'] = tl.str.contains('جديد|جديدة|neuf', na=False).astype(int)
    df['is_luxury'] = tl.str.contains('لويكس|luxe|راقي|فاخر', na=False).astype(int)
    df['is_two_fl'] = tl.str.contains('طابقين', na=False).astype(int)
    df['is_opport'] = tl.str.contains('فرصة|فرصه', na=False).astype(int)
    df['kw_price'] = tl.str.contains('السعر|مبيوع ب', na=False).astype(int)
    df['kw_road'] = tl.str.contains('اعل شارع|على شارع', na=False).astype(int)
    df['kw_neuf'] = tl.str.contains('جديد|مافات', na=False).astype(int)
    
    # Features numériques
    df['nb_pieces'] = df['nb_chambres'] + df['nb_salons'] + df['nb_sdb']
    df['surf_p_pce'] = df['surface_m2'] / (df['nb_pieces'] + 1)
    df['ch_per_sdb'] = df['nb_chambres'] / (df['nb_sdb'] + 1)
    df['log_surf'] = np.log1p(df['surface_m2'])
    df['surf_sq'] = df['surface_m2'] ** 2
    df['surf_bucket'] = pd.cut(df['surface_m2'],
        bins=[0, 100, 150, 200, 300, 500, 1000, float('inf')],
        labels=[0, 1, 2, 3, 4, 5, 6]).astype(float).fillna(0)
    df['log_taille_rue'] = np.log1p(df['taille_rue'])
    df['desc_len'] = df['description'].str.len()
    df['titre_len'] = df['titre'].str.len()
    
    dt = pd.to_datetime(df['date_publication'], errors='coerce')
    df['pub_month'] = dt.dt.month.fillna(6).astype(int)
    df['pub_quarter'] = dt.dt.quarter.fillna(2).astype(int)
    df['pub_year'] = dt.dt.year.fillna(2025).astype(int)
    df['source_enc'] = 0
    
    # Interactions
    df['villa_surf'] = df['is_villa'] * df['surface_m2']
    df['duplex_surf'] = df['is_duplex'] * df['surface_m2']
    df['land_surf'] = df['is_land'] * df['surface_m2']
    df['corner_surf'] = df['is_corner'] * df['surface_m2']
    df['surf_x_carac'] = df['surface_m2'] * df['nb_carac']
    df['log_surf_x_np'] = df['log_surf'] * df['nb_pieces']
    
    # TF-IDF
    tfidf_mat = tfidf.transform(df['text_all']).toarray()
    for i, col in enumerate(tfidf_cols):
        df[col] = tfidf_mat[:, i]
    
    # Proche de
    df['proche_de'] = df['proche_raw'].apply(lambda x: x if x in top_proche else 'other')
    
    # Target encoding
    q = df['quartier'].iloc[0]
    df['quartier_enc'] = quartier_te.get(q, global_mean)
    df['proche_de_enc'] = proche_te.get(df['proche_de'].iloc[0], global_mean)
    df['q_ppm2'] = q_ppm2_map.get(q, q_ppm2_global)
    
    return df


# --- Endpoint de prédiction ---
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON body requis'}), 400
        
        # Feature engineering
        df = engineer_single(data)
        
        # Extraire les features pour le modèle structurel
        feat_cols = FEATS + ['quartier_enc', 'proche_de_enc', 'q_ppm2']
        X = np.nan_to_num(df[feat_cols].values.astype(np.float64), nan=0.0)
        
        # Prédictions des 3 modèles
        pred_lgb = lgb_model.predict(X)[0]
        pred_xgb = xgb_model.predict(X)[0]
        pred_cb = cb_model.predict(X)[0]
        
        # Ensemble pondéré
        pred_struct = w_lgb * pred_lgb + w_xgb * pred_xgb + w_cb * pred_cb
        
        # MoE : mélange avec text_price si disponible
        has_price = df['has_price'].iloc[0]
        text_price = df['text_price'].iloc[0]
        
        if has_price == 1 and text_price > 0:
            text_price_log = np.log1p(text_price)
            pred_final_log = best_alpha * text_price_log + (1 - best_alpha) * pred_struct
        else:
            pred_final_log = pred_struct
        
        prix_estime = int(np.expm1(pred_final_log))
        
        return jsonify({
            'prix_estime': prix_estime,
            'devise': 'MRU',
            'details': {
                'pred_structurel': int(np.expm1(pred_struct)),
                'text_price_detecte': int(text_price) if has_price else None,
                'alpha_moe': best_alpha,
                'methode': 'MoE (text + struct)' if has_price else 'Structurel seul'
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'nouakchott_v7', 'features': len(FEATS)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)