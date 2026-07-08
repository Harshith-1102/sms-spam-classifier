# ============================================================
#  SMS Spam Classifier — Full ML Pipeline
#  Dataset: SMSSpamCollection.csv (tab-separated)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve,
                              roc_curve, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp

# ─────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a1a',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#999999',
    'ytick.color':      '#999999',
    'text.color':       '#cccccc',
    'grid.color':       '#2a2a2a',
    'grid.linestyle':   '--',
    'font.family':      'monospace',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

COLORS = {
    'spam':    '#e05c3a',
    'ham':     '#3a9fd5',
    'accent':  '#b88aff',
    'success': '#3acf8a',
    'warn':    '#f0b429',
    'line':    '#444444',
}

STOP = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv('/mnt/user-data/uploads/SMSSpamCollection.csv',
                 sep='\t', header=None, names=['label', 'message'], quoting=3)
df['label'] = df['label'].str.strip().str.replace('"', '')
df = df.dropna().reset_index(drop=True)

print(f"Total messages  : {len(df)}")
print(f"Ham             : {(df['label']=='ham').sum()}")
print(f"Spam            : {(df['label']=='spam').sum()}")
print(f"Missing values  : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Feature engineering")
print("=" * 60)

df['msg_length']       = df['message'].apply(len)
df['word_count']       = df['message'].apply(lambda x: len(x.split()))
df['num_digits']       = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
df['num_special']      = df['message'].apply(lambda x: sum(c in string.punctuation for c in x))
df['uppercase_ratio']  = df['message'].apply(
    lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
df['has_url']          = df['message'].str.contains(
    r'http[s]?://|www\.|\bclick\b|\bfree\b|\bwon\b|\bprize\b', case=False).astype(int)
df['exclaim_count']    = df['message'].str.count(r'!')
df['currency_count']   = df['message'].str.count(r'[£$€]')

print("Engineered features:")
feat_cols = ['msg_length','word_count','num_digits','num_special',
             'uppercase_ratio','has_url','exclaim_count','currency_count']
print(df.groupby('label')[feat_cols].mean().round(3).to_string())

# ─────────────────────────────────────────
# 3. TEXT CLEANING
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — Text cleaning & stemming")
print("=" * 60)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in STOP and len(w) > 1]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(clean_text)
print("Sample cleaned messages:")
for _, row in df.sample(3, random_state=42).iterrows():
    print(f"  [{row['label']}] {row['message'][:60]}")
    print(f"       → {row['clean_message'][:60]}\n")

# ─────────────────────────────────────────
# 4. EDA PLOTS (saved)
# ─────────────────────────────────────────
print("=" * 60)
print("  STEP 4 — EDA visualizations")
print("=" * 60)

fig = plt.figure(figsize=(16, 12), facecolor='#0f0f0f')
fig.suptitle('SMS Spam Collection — EDA', color='white', fontsize=16, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 4a — Class distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = df['label'].value_counts()
bars = ax0.bar(counts.index, counts.values,
               color=[COLORS['ham'], COLORS['spam']], width=0.5, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom',
             color='white', fontsize=10)
ax0.set_title('Class distribution')
ax0.set_ylim(0, counts.max() * 1.2)
ax0.grid(axis='y')

# 4b — Message length distribution
ax1 = fig.add_subplot(gs[0, 1])
for label, color in [('ham', COLORS['ham']), ('spam', COLORS['spam'])]:
    ax1.hist(df[df['label']==label]['msg_length'], bins=40,
             alpha=0.6, label=label, color=color, edgecolor='none')
ax1.set_title('Message length')
ax1.set_xlabel('Characters')
ax1.legend()
ax1.grid(axis='y')

# 4c — Word count
ax2 = fig.add_subplot(gs[0, 2])
for label, color in [('ham', COLORS['ham']), ('spam', COLORS['spam'])]:
    ax2.hist(df[df['label']==label]['word_count'], bins=30,
             alpha=0.6, label=label, color=color, edgecolor='none')
ax2.set_title('Word count')
ax2.set_xlabel('Words')
ax2.legend()
ax2.grid(axis='y')

# 4d — Uppercase ratio boxplot
ax3 = fig.add_subplot(gs[1, 0])
ham_data  = df[df['label']=='ham']['uppercase_ratio']
spam_data = df[df['label']=='spam']['uppercase_ratio']
bp = ax3.boxplot([ham_data, spam_data], labels=['ham','spam'],
                 patch_artist=True, medianprops={'color':'white','linewidth':2})
bp['boxes'][0].set_facecolor(COLORS['ham'])
bp['boxes'][1].set_facecolor(COLORS['spam'])
for element in ['whiskers','caps','fliers']:
    for item in bp[element]:
        item.set_color('#666666')
ax3.set_title('Uppercase ratio')
ax3.grid(axis='y')

# 4e — Feature means comparison
ax4 = fig.add_subplot(gs[1, 1])
compare_feats = ['num_digits','num_special','exclaim_count','currency_count']
ham_means  = df[df['label']=='ham'][compare_feats].mean()
spam_means = df[df['label']=='spam'][compare_feats].mean()
x = np.arange(len(compare_feats))
w = 0.35
ax4.bar(x - w/2, ham_means,  width=w, label='ham',  color=COLORS['ham'],  edgecolor='none')
ax4.bar(x + w/2, spam_means, width=w, label='spam', color=COLORS['spam'], edgecolor='none')
ax4.set_xticks(x)
ax4.set_xticklabels(['digits','special','!','currency'], fontsize=9)
ax4.set_title('Feature means: ham vs spam')
ax4.legend()
ax4.grid(axis='y')

# 4f — URL presence
ax5 = fig.add_subplot(gs[1, 2])
url_counts = df.groupby(['label','has_url']).size().unstack(fill_value=0)
url_counts.T.plot(kind='bar', ax=ax5, color=[COLORS['ham'], COLORS['spam']],
                  edgecolor='none', legend=True)
ax5.set_xticklabels(['No URL','Has URL'], rotation=0)
ax5.set_title('URL presence')
ax5.grid(axis='y')

plt.savefig('/home/claude/eda.png', dpi=130, bbox_inches='tight',
            facecolor='#0f0f0f')
plt.close()
print("EDA plot saved → eda.png")

# ─────────────────────────────────────────
# 5. PREPARE DATA
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — Vectorization & train/test split")
print("=" * 60)

le = LabelEncoder()
y = le.fit_transform(df['label'])   # ham=0, spam=1

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_text = tfidf.fit_transform(df['clean_message'])

hand_feats = df[feat_cols].values
X_combined = sp.hstack([X_text, sp.csr_matrix(hand_feats)])

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")
print(f"Features   : {X_combined.shape[1]} (TF-IDF bigrams + 8 hand-crafted)")

# ─────────────────────────────────────────
# 6. TRAIN & COMPARE MODELS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — Model training & comparison")
print("=" * 60)

models = {
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced',
                                               max_iter=1000, random_state=42),
    'Linear SVM':          LinearSVC(C=0.5, class_weight='balanced',
                                     max_iter=2000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                   class_weight='balanced',
                                                   random_state=42, n_jobs=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv,
                            scoring='f1', n_jobs=-1).mean()

    results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'cv_f1':     cv_f1,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'model':     model,
    }
    print(f"\n{name}")
    print(f"  Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"  F1        : {results[name]['f1']:.4f}")
    print(f"  ROC-AUC   : {results[name]['roc_auc']:.4f}")
    print(f"  CV F1     : {results[name]['cv_f1']:.4f}")

# ─────────────────────────────────────────
# 7. EVALUATION PLOTS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7 — Evaluation plots")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='#0f0f0f')
fig.suptitle('Model Evaluation — SMS Spam Classifier', color='white',
             fontsize=15, y=0.99)

pal = [COLORS['ham'], COLORS['spam'], COLORS['accent'], COLORS['warn']]
model_names = list(results.keys())

# 7a — Accuracy comparison
ax = axes[0, 0]
vals = [results[n]['accuracy'] for n in model_names]
bars = ax.barh(model_names, vals, color=pal, edgecolor='none', height=0.5)
for bar, val in zip(bars, vals):
    ax.text(val - 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right', color='white', fontsize=10)
ax.set_xlim(0.9, 1.01)
ax.set_title('Accuracy')
ax.grid(axis='x')

# 7b — F1 Score
ax = axes[0, 1]
vals = [results[n]['f1'] for n in model_names]
bars = ax.barh(model_names, vals, color=pal, edgecolor='none', height=0.5)
for bar, val in zip(bars, vals):
    ax.text(val - 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right', color='white', fontsize=10)
ax.set_xlim(0.8, 1.01)
ax.set_title('F1 Score (spam class)')
ax.grid(axis='x')

# 7c — ROC-AUC
ax = axes[0, 2]
vals = [results[n]['roc_auc'] for n in model_names]
bars = ax.barh(model_names, vals, color=pal, edgecolor='none', height=0.5)
for bar, val in zip(bars, vals):
    ax.text(val - 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right', color='white', fontsize=10)
ax.set_xlim(0.9, 1.01)
ax.set_title('ROC-AUC')
ax.grid(axis='x')

# 7d — Confusion matrix for best model
best_name = max(results, key=lambda n: results[n]['f1'])
ax = axes[1, 0]
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['ham','spam'], yticklabels=['ham','spam'],
            linewidths=0.5, linecolor='#333333',
            annot_kws={'size': 14, 'color': 'white'})
ax.set_title(f'Confusion matrix — {best_name}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# 7e — ROC Curves
ax = axes[1, 1]
for name, color in zip(model_names, pal):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color, linewidth=1.8)
ax.plot([0,1],[0,1], '--', color='#555555', linewidth=1)
ax.set_title('ROC curves')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8, facecolor='#222222', edgecolor='#444444')
ax.grid()

# 7f — Precision-Recall for best model
ax = axes[1, 2]
prec, rec, thr = precision_recall_curve(y_test, results[best_name]['y_prob'])
ax.plot(rec, prec, color=COLORS['spam'], linewidth=2)
ax.fill_between(rec, prec, alpha=0.15, color=COLORS['spam'])
ax.set_title(f'Precision-Recall — {best_name}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid()

plt.tight_layout()
plt.savefig('/home/claude/evaluation.png', dpi=130, bbox_inches='tight',
            facecolor='#0f0f0f')
plt.close()
print(f"Evaluation plots saved → evaluation.png")
print(f"Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

# ─────────────────────────────────────────
# 8. FULL CLASSIFICATION REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  STEP 8 — Detailed report: {best_name}")
print("=" * 60)
print(classification_report(y_test, results[best_name]['y_pred'],
                             target_names=['ham','spam']))

# ─────────────────────────────────────────
# 9. TOP TF-IDF FEATURES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 9 — Top spam-indicating words (TF-IDF)")
print("=" * 60)

# Refit Naive Bayes on text-only for interpretability
tfidf_only = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_text_only = tfidf_only.fit_transform(df['clean_message'])
nb_interp = MultinomialNB(alpha=0.1)
nb_interp.fit(X_text_only, y)

feat_names = np.array(tfidf_only.get_feature_names_out())
log_probs = nb_interp.feature_log_prob_[1] - nb_interp.feature_log_prob_[0]
top_idx = np.argsort(log_probs)[-15:][::-1]

print("Top 15 spam-indicating terms:")
for i, idx in enumerate(top_idx, 1):
    print(f"  {i:2}. {feat_names[idx]:<20} score={log_probs[idx]:.3f}")

# ─────────────────────────────────────────
# 10. PREDICT NEW MESSAGES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 10 — Live prediction demo")
print("=" * 60)

best_model = results[best_name]['model']

def predict_message(msg):
    cleaned = clean_text(msg)
    tf = tfidf.transform([cleaned])
    feats = [[
        len(msg),
        len(msg.split()),
        sum(c.isdigit() for c in msg),
        sum(c in string.punctuation for c in msg),
        sum(c.isupper() for c in msg) / max(len(msg), 1),
        int(bool(re.search(r'http|www|free|click|won|prize', msg, re.I))),
        msg.count('!'),
        len(re.findall(r'[£$€]', msg))
    ]]
    x = sp.hstack([tf, sp.csr_matrix(feats)])
    pred = best_model.predict(x)[0]
    label = le.inverse_transform([pred])[0]
    return label.upper()

test_messages = [
    "Congratulations! You've won a £1000 prize. Call now FREE on 0800123456!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Click here to verify now!",
    "Can you pick up some milk on your way home?",
    "FREE entry to win a holiday! Text WIN to 87121. T&Cs apply.",
]

for msg in test_messages:
    result = predict_message(msg)
    icon = "🚨" if result == "SPAM" else "✅"
    print(f"  {icon} [{result}] {msg[:65]}")

print("\n" + "=" * 60)
print("  Pipeline complete!")
print("=" * 60)
