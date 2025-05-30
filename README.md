# BP: Webová aplikácia pre predikciu a interpretáciu medicínskych a finančných dát

Repozitár obsahuje bakalársky projekt zameraný na spracovanie, predikciu a interpretáciu dát z medicínskej a finančnej oblasti. Projekt je implementovaný vo Flasku (Python) s využitím hlbokého učenia a metód vysvetliteľnosti (XAI).

## 🧠 Ciele projektu

- Implementovať a zlepsit vysvetliteľnosť modelu cez LIME (Local Interpretable Model-agnostic Explanations)
- Využiť viacvrstvové neurónové siete (deep learning) na klasifikáciu a regresiu
- Predikovať výskyt cukrovky (Diabetes), rakoviny (Cancer) a zdravotné poistenie (Insurance)
- Vytvoriť intuitívne a interaktívne webové rozhranie

## 📁 Štruktúra projektu

```
.
├── app.py                 # Backend aplikácie (Flask)
├── deep_models.py         # Tréning modelov (TensorFlow, Keras)
├── models/                # Vytrénované modely, scalery, encodery
├── datasets/              # CSV súbory s dátami
├── templates/             # HTML šablóny (Jinja2)
├── static/                # CSS štýly (Bootstrap)
├── requirements.txt       # Zoznam závislostí
└── README.md              # Tento súbor
```

## ⚙️ Inštalácia

1. Klonovanie repozitára:

```bash
git clone https://github.com/morresmx/BP.git
cd BP
```

2. Vytvorenie virtuálneho prostredia:

```bash
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate
```

3. Inštalácia závislostí:

```bash
pip install -r requirements.txt
```

## 🚀 Spustenie

Projekt je určený na spustenie v prostredí Jupyter Notebook. Po aktivácii prostredia spustite:

```bash
python app.py
```

Aplikácia bude dostupná na http://127.0.0.1:5000/

## 📊 Použité technológie

- Python 3.11+
- Flask
- TensorFlow, Keras
- scikit-learn
- lime
- pandas, numpy, matplotlib
- Bootstrap 5

## 💡 Použité datasety

- [Diabetes](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- [Cancer](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)
- [Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
