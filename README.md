# Smart Medical Chatbot

An AI-powered medical chatbot that provides real-time symptom triage and personalized health guidance via a conversational interface.

## 🚀 Features
- **24/7 Conversational Care:** Handles free-form user queries and provides context-aware medical advice  
- **Symptom Checker Integration:** Cross-references reported symptoms via an external API for accurate suggestions  
- **Multi-Language Support:** Communicates in multiple languages to serve diverse users  
- **Web UI:** Responsive React.js + Tailwind CSS interface for seamless interactions  
- **Secure & Compliant:** Follows best practices for data privacy (no PHI storage)

## 🛠️ Prerequisites
- **Node.js** ≥ 14.x  
- **Python** ≥ 3.8  
- A free API key for your chosen symptom-checker service  

## ⚙️ Installation

1. **Clone the repo**  
   ```
   git clone https://github.com/utkarshagawade17/SmartMedibot.git
   cd SmartMedibot
   ```
2. Backend setup
   ```
   cd backend
   python3 -m venv env
   source env/bin/activate
   # on Windows: `.\env\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Frontend setup
```
cd ../frontend
npm install
```
Running Locally
Start the backend
```
cd backend
source env/bin/activate
python app.py
```
Start the frontend
```
cd ../frontend
npm run dev
```
Open your browser at http://localhost:3000

Usage:
Type in your symptoms and hit “Submit” — the chatbot will respond with possible conditions and next-step advice.
