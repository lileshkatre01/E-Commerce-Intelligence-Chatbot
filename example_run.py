"""Example script demonstrating standalone chatbot usage without Jupyter.

Flow:
  Question
     ↓
  retrieve_reviews()
     ↓
  Existing FAISS embeddings
     ↓
  Top relevant reviews
     ↓
  Gemini
     ↓
  Answer
"""

from chatbot import ask_chatbot

if __name__ == "__main__":
    print("Sending question to chatbot...\n")
    answer = ask_chatbot("What are the major complaints from customers?")
    print("=" * 60)
    print("CHATBOT RESPONSE:")
    print("=" * 60)
    print(answer)
