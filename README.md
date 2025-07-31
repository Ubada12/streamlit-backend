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

1. **Download the model** from [this link](https://drive.google.com/uc?export=download&id=1AU0MuvewEeXWVu-j-WoXO6bw391VGyNl).  
2. Move the downloaded file to the `models/` directory.  

Alternatively, you can use the following command:  
```bash
if ((Split-Path -Leaf (Get-Location)) -ne "streamlit-backend") {
    Write-Host "❌ Please navigate to the 'streamlit-backend' directory before running this command." -ForegroundColor Red
} else {
    $url="https://drive.google.com/uc?export=download&id=1AU0MuvewEeXWVu-j-WoXO6bw391VGyNl"
    $zip="ml_models.zip"
    Invoke-WebRequest $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath .
    Remove-Item $zip
    if (Test-Path ".\ml_models") {
        Write-Host "✅ ml_models folder successfully added!" -ForegroundColor Green
    } else {
        Write-Host "❌ Extraction failed. Check ZIP contents." -ForegroundColor Red
    }
}

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
