import os
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it in .env file.")

# Load AI Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# AI Prompt for Recommendations
product_recommendation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant for an e-commerce platform. Recommend {count} products from the {category} category under {budget}."),
        ("human", "Suggest {count} products.")
    ]
)

# Format Response
format_response = RunnableLambda(lambda x: f"📦 Recommended Products:\n\n{x}")

# AI Chain
chain = product_recommendation_template | model | StrOutputParser() | format_response

# Function to get recommendations
def recommend_products(category, budget, count):
    try:
        result = chain.invoke({"category": category, "budget": budget, "count": int(count)})
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=recommend_products,
    inputs=[
        gr.Textbox(label="Product Category", placeholder="laptops, smartphones, etc."),
        gr.Textbox(label="Budget", placeholder="$500, $1000, etc."),
        gr.Number(label="Number of Recommendations", value=3)
    ],
    outputs="text",
    title="🛒 E-commerce Product Recommendation Bot",
    description="Enter product category, budget, and number of recommendations to get AI-based product suggestions.",
)

# Run Locally
if __name__ == "__main__":
    iface.launch()
