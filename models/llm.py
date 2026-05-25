from groq import Groq
from dotenv import load_dotenv

import os

# ---------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------

load_dotenv()

# ---------------------------------------------
# GROQ CLIENT
# ---------------------------------------------

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------------------------
# MODEL NAME
# ---------------------------------------------

MODEL_NAME = "llama-3.1-8b-instant"


# ---------------------------------------------
# GENERATE ANSWER FUNCTION
# ---------------------------------------------

def generate_answer(question, retrieved_chunks):

    try:

        # -----------------------------------------
        # COMBINE CONTEXT
        # -----------------------------------------

        context = "\n\n".join(retrieved_chunks)

        # -----------------------------------------
        # PROMPT
        # -----------------------------------------

        prompt = f"""
        You are Lumina AI,
        an intelligent AI learning assistant.

        Use ONLY the provided context
        to answer the question.

        Rules:
        - Give clear and concise answers
        - Use student-friendly language
        - If answer is unavailable,
          say:
          'Answer not found in uploaded notes.'

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # -----------------------------------------
        # GROQ RESPONSE
        # -----------------------------------------

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",
                    "content":
                    "You are a helpful AI study assistant."
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=700
        )

        # -----------------------------------------
        # RETURN RESPONSE
        # -----------------------------------------

        return response.choices[0].message.content

    except Exception as e:

        return f"Error: {str(e)}"