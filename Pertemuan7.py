# Pertemuan 7 
# Langkah 1 — Siapkan Data
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_kelulusan.csv")
x = df.drop("Lulus", axis=1)
y = df['Lulus']

sc = StandardScaler()
Xs =sc.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(x_train.shape, x_val.shape, x_test.shape)
print("\n==========Batas antar Output==========\n")

# Langkah 2 - Bangun model ANN 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # klasifikasi biner
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])
model.summary()
print("\n==========Batas antar Output==========\n")

# langkah 3 
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es], verbose=1
)
print("\n==========Batas antar Output==========\n")

# langkah 4 
from sklearn.metrics import classification_report, confusion_matrix

loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
# print("Test Acc:", acc, "AUC:", auc)

y_proba = model.predict(x_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("\n==========Batas antar Output==========\n")

# langkah 5
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve")
plt.tight_layout(); plt.savefig("learning_curve.png", dpi=120)

# langkah 6 
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
import pandas as pd

# Variasi jumlah neuron, optimizer, dan regularisasi
experiments = [
    {"neurons": 32, "optimizer": "adam", "dropout": 0.3, "l2": 0.0, "batch_norm": False},
    {"neurons": 64, "optimizer": "adam", "dropout": 0.3, "l2": 0.0, "batch_norm": False},
    {"neurons": 128, "optimizer": "adam", "dropout": 0.3, "l2": 0.0, "batch_norm": False},
    {"neurons": 64, "optimizer": SGD(learning_rate=0.01, momentum=0.9), "dropout": 0.3, "l2": 0.0, "batch_norm": False},
    {"neurons": 64, "optimizer": "adam", "dropout": 0.5, "l2": 0.01, "batch_norm": True},
]

results = []

for i, exp in enumerate(experiments, 1):
    print(f"\n--- Eksperimen {i}: {exp} ---\n")
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(x_train.shape[1],)))
    
    if exp["batch_norm"]:
        model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(exp["neurons"], activation="relu",
                           kernel_regularizer=regularizers.l2(exp["l2"])))
    model.add(layers.Dropout(exp["dropout"]))
    
    if exp["batch_norm"]:
        model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(exp["neurons"]//2, activation="relu",
                           kernel_regularizer=regularizers.l2(exp["l2"])))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=exp["optimizer"],
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )
    
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100, batch_size=32,
        callbacks=[es], verbose=0
    )
    
    # Evaluasi
    loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
    y_proba = model.predict(x_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Experiment": i,
        "Neurons": exp["neurons"],
        "Optimizer": type(exp["optimizer"]).__name__ if not isinstance(exp["optimizer"], str) else exp["optimizer"],
        "Dropout": exp["dropout"],
        "L2": exp["l2"],
        "BatchNorm": exp["batch_norm"],
        "Accuracy": acc,
        "AUC": auc,
        "F1": f1
    })

# Tampilkan hasil eksperimen
results_df = pd.DataFrame(results)
print("\nHasil Eksperimen:\n")
print(results_df)


