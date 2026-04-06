import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig("Results/confusion_matrix.png")

def plot_confidence(confidences):
    plt.hist(confidences, bins=20)
    plt.title("Prediction Confidence Distribution")
    plt.savefig("Results/confidence_distribution.png")
