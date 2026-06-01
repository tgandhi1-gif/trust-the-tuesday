"""Trust the Tuesday - does one first week action predict trial churn?"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
N = 8000

core_action = rng.random(N) < 0.45
logins = rng.poisson(3, N)
invited_team = rng.random(N) < 0.30

p_churn = 0.80 - 0.50 * core_action - 0.05 * np.minimum(logins, 5) / 5 - 0.10 * invited_team
p_churn = np.clip(p_churn, 0.05, 0.95)
churn = rng.random(N) < p_churn

X = pd.DataFrame({
    "core_action": core_action.astype(int),
    "logins": logins,
    "invited_team": invited_team.astype(int),
})
y = churn.astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
model = GradientBoostingClassifier().fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
lift = churn[~core_action].mean() / churn[core_action].mean()

print(f"Churn model AUC: {auc:.3f}")
print(f"Users who skipped the first week action churn {lift:.1f}x more")

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.barh(imp.index, imp.values, color="#ff6a3d")
plt.title("What predicts churn (first week signals)")
plt.tight_layout()
plt.savefig("outputs/trust_the_tuesday.png", dpi=120)
print("Saved outputs/trust_the_tuesday.png")
