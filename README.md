---

## **Flood Predictor 🌊**  

A deep learning-based flood prediction model using VGG16.  

---

### **📌 Prerequisites**  
Ensure you have the following installed before running the project:  

- Python 3.8+  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

### **📥 Download Pretrained Model**  
The pretrained VGG16 model is required to run predictions.  

1. **Download the model** from [this link](https://drive.google.com/file/d/1B6NoDc4ejFL4ogkzpIOb4dbLpiSngpSn/view?usp=drive_link).  
2. Move the downloaded file to the `models/` directory.  

Alternatively, you can use the following command:  
```bash
wget "https://drive.google.com/uc?export=download&id=1B6NoDc4ejFL4ogkzpIOb4dbLpiSngpSn" -O models/vgg16_model.keras
```

---

### **🚀 Usage**  
Run fast API server:  
```bash
unicorn main:app
```
Run streamlit server:
```bash
streamlit run client_main.py
```

---

### **📜 License**  
This project is licensed under the MIT License.  

---

Let me know if you need any modifications! 🚀
