import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("stress.csv")

# Split into 3 clients
client1 = data.sample(frac=0.33, random_state=1)
remaining = data.drop(client1.index)

client2 = remaining.sample(frac=0.5, random_state=2)
client3 = remaining.drop(client2.index)

clients = [client1, client2, client3]

models = []

for client in clients:
    X = client.drop("Stress", axis=1)
    y = client["Stress"]

    model = RandomForestClassifier()
    model.fit(X, y)

    models.append(model)

# Simple aggregation (choose best model)
global_model = models[0]

# Save model
pickle.dump(global_model, open("model.pkl","wb"))

print("Federated Global Model Created")