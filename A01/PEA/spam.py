import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paso 1: Cargar el dataset
# Para este ejemplo, usaremos un dataset conocido de spam como el "SMS Spam Collection".
# Supongamos que el archivo `spam.csv` contiene las columnas ["label", "message"].

# Cargar los datos
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "message"})
df = df[["label", "message"]]

# Convertir etiquetas en valores binarios
df['label'] = df['label'].map({'spam': 1, 'ham': 0})  # spam = 1, ham = 0

# Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Vectorización del texto con TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Paso 4: Entrenar el modelo (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Paso 5: Evaluar el modelo
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Exactitud del modelo: {accuracy:.2f}")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Paso 6: Hacer predicciones con nuevos datos
new_emails = ["Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now.",
              "Hi, just wanted to follow up on the meeting tomorrow."]
new_emails_tfidf = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_tfidf)

for email, label in zip(new_emails, predictions):
    print(f"Email: {email}\nPredicción: {'Spam' if label == 1 else 'No Spam'}\n")
