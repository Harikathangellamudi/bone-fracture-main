# 🦴 Bone Fracture Detection using ResNet50

This project implements a **deep learning-based bone fracture detection system** using **ResNet50** architecture.  
It is designed to classify **X-ray images** into **fractured** or **normal** categories, helping in **automated medical diagnosis** and reducing manual workload.  
The system includes a **Graphical User Interface (GUI)** built with **Streamlit** for easy interaction.

---

## 🚀 Features
- **Model Architecture**: ResNet50 (pretrained on ImageNet, fine-tuned for medical imaging).
- **Activation Functions**:
  - **ReLU** → used in intermediate layers for non-linearity.
  - **Softmax** → used in the output layer for classification probabilities.
- **Optimizer**: Adam optimizer.
- **Learning Rate**: `0.0001` with dynamic adjustments.
- **Training Enhancements**:
  - **Early Stopping** → prevents overfitting by stopping training when validation accuracy plateaus.
  - **Data Augmentation** → improves generalization for unseen X-rays.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- **Streamlit GUI** → for uploading and classifying bone X-rays.

---

## 📂 Project Structure
