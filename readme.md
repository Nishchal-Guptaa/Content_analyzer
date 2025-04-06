# 🧠 Toxic Content Classifier Web App

A Flask-based web application that allows users to **register**, **log in**, and **analyze text content** for toxicity, NSFW, and spam using a BERT-based model from Hugging Face 🤖.

---

## 🔍 Features

- 🔐 User Authentication (Login, Signup, Logout)
- 📊 Toxic comment classification using [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)
- 🧠 Detects multiple types of toxicity:
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`
- 🔞 NSFW keyword detection
- 🚫 Spam keyword detection
- 📡 JSON-based API response for classification

---

## 🛠️ Tech Stack

- **Backend**: Flask, SQLAlchemy, Flask-Login, Flask-WTF
- **Model**: Transformers (`unitary/toxic-bert`), PyTorch
- **Database**: SQLite
- **Security**: Password hashing with Flask-Bcrypt
- **Environment Management**: `python-dotenv`

---

## 🖥️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/toxic-content-classifier.git
cd toxic-content-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory:

```bash
SECRET_KEY=your_secret_key_here
```

### 5. Initialize the Database

```bash
python
>>> from app import db
>>> db.create_all()
>>> exit()
```

### 6. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

---

## 📬 API Endpoint

**POST** `/classify`

**Request:**
```json
{
  "text": "You are a horrible person!"
}
```

**Response:**
```json
{
  "comment": "You are a horrible person!",
  "prediction_tags": ["toxic", "insult"],
  "toxic_scores": {
    "toxic": 89.23,
    "severe_toxic": 4.11,
    "obscene": 22.91,
    "threat": 0.34,
    "insult": 75.56,
    "identity_hate": 1.02
  }
}
```

---

## 📁 Project Structure

```
.
├── app.py
├── database.db
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── model.html
│   └── sign_up.html
├── static/
├── .env
└── requirements.txt
```

---

## 🧪 Example Keywords

- NSFW: `xxx`, `hotgirls`, `sex`, `nude`
- Spam: `click here`, `win`, `free`, `subscribe`, `giveaway`

---

## 🙌 Credits

- [Unitary Toxic BERT](https://huggingface.co/unitary/toxic-bert)
- Flask Community
- Hugging Face Transformers

---

## 🛡 Disclaimer

This project is for educational purposes. The model may not be fully accurate and should not be used in production without further evaluation.

---

Let me know if you want a logo, badges (build passing, license, etc.), or deployment instructions (like on Render or Heroku)!
