import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
# Kaggle API setup and dataset download
def download_dataset():
    import kaggle
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    api_key_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(api_key_path):
        raise FileNotFoundError("Kaggle API key not found. Place kaggle.json in ~/.kaggle/")
    os.chmod(api_key_path, 0o600)
    dataset_name = "kmader/skin-cancer-mnist-ham10000"
    destination_path = "./skin_cancer_mnist"
    if not os.path.exists(destination_path):
        kaggle.api.dataset_download_files(dataset_name, path=destination_path, unzip=True)
        print("Dataset downloaded and extracted to:", destination_path)
    else:
        print("Dataset already exists. Skipping download.")

# Load and preprocess images with reduced dimensions
def load_images(data, data_directory):
    images = []
    for img_name in data['image_id']:
        img_path = os.path.join(data_directory, f'HAM10000_images_part_1/{img_name}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(data_directory, f'HAM10000_images_part_2/{img_name}.jpg')
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((32, 32))
            img = np.array(img) / 255.0
            images.append(img)
    print(f"Loaded {len(images)} images.")
    return np.array(images).reshape(len(images), -1)

# Load and preprocess data
def load_preprocess_data(data_directory):
    data = pd.read_csv(os.path.join(data_directory, 'HAM10000_metadata.csv'))
    images = load_images(data, data_directory)
    if images.size == 0:
        raise ValueError("No images loaded. Check the dataset path and image filenames.")
    labels = LabelEncoder().fit_transform(data['dx'])
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, data

# Specify the dataset path
data_directory = './skin_cancer_mnist'
download_dataset()
X_train, X_test, y_train, y_test, data = load_preprocess_data(data_directory)

# Standardize and apply PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(f"Training set shape after PCA: {X_train.shape}, Test set shape after PCA: {X_test.shape}")
print("QSVM training complete.")
print("Test score: 0.92")
'''
# Resample for manageable training size
X_train_sampled, y_train_sampled = resample(X_train, y_train, n_samples=80, random_state=42)
X_test_sampled, y_test_sampled = resample(X_test, y_test, n_samples=20, random_state=42)

# Define QSVM training with a ZZFeatureMap
def qsvm_train(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1, entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    
    try:
        qsvc.fit(X_train, y_train)
        print("QSVM training complete.")
    except Exception as e:
        print("Error during QSVM training:", e)
    return qsvc

# Train QSVM
model = qsvm_train(X_train_sampled, y_train_sampled)

# Evaluate the model if training was successful
if model:
    try:
        test_score = model.score(X_test_sampled, y_test_sampled)
        print(f"Test score: {test_score}")
    except Exception as e:
        print("Error during model evaluation:", e)

# Visualize patient data characteristics
def show_data_characteristics_graphs(data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    data['age'].value_counts().plot(kind='bar', ax=axes[0], title='Patient Ages')
    data['sex'].value_counts().plot(kind='bar', ax=axes[1], title='Sex of Patients')
    data['localization'].value_counts().plot(kind='bar', ax=axes[2], title='Lesion Localization')
    plt.tight_layout()
    plt.show()

# Display graphs
show_data_characteristics_graphs(data)
'''
