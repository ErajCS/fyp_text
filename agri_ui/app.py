from flask import Flask, render_template, request, jsonify
from rag_demo import rag_pipeline # Import your existing function

app = Flask(__name__)

# 1. Serve the HTML Interface
@app.route("/")
def home():
    return render_template("chatbot.html")

# 2. Handle the Chat Logic
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("msg")
    
    # Call your existing RAG pipeline
    # This runs your retrieval + generation logic
    ai_response = rag_pipeline(user_message)
    
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    # Open automatically in the browser
    print("Starting AgriChat Interface...")
    app.run(debug=True)