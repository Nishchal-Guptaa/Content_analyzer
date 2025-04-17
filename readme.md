# 🛡️ AI-Powered Content Moderation Web App

This is a full-featured content moderation application built using Flask. It uses powerful transformer models to detect toxic text, NSFW content in images, and spam or inappropriate keywords in uploaded files. The app supports user authentication, admin analytics, and classification history tracking.

---

## 🚀 Features

- 🔐 **User authentication** with registration and login
- 🧠 **Text classification** using BERT (`unitary/toxic-bert`)
- 🖼️ **Image classification** using ViT (`Falconsai/nsfw_image_detection`)
- 📄 **File (.txt) moderation** with chunked text analysis
- 🗂️ **Classification history** saved per user
- 👨‍💼 **Admin dashboard** for stats and user management (via Flask-Admin)
- 🌐 RESTful API routes for moderation

---

## 🧰 Tech Stack

- Flask
- SQLAlchemy + SQLite
- Flask-Login, Flask-WTF, Flask-Admin
- Transformers (HuggingFace)
- Torch (PyTorch)
- PIL (for image handling)
- dotenv (for secure config)

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
touch .env
```
In your .env file
```
SECRET_KEY=your-secret-key-here
```
