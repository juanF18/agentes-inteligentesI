import pickle

# Ruta al archivo .pkl
file_path = "./TD-Learning/QLunarTD0Softmax.pkl"

# Abrir el archivo .pkl y cargar los datos
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Imprimir la primera línea del archivo
print("Primera línea del archivo .pkl:")
# print(len(data[0][0][0][0][0][0][0][0]))
print(data[0][0][0][0][0][0][0])
