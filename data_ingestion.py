from pathlib import Path
import pandas as pd
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
import pickle

# Define a function to load a DataFrame from a specified file path
def load_dataframe(file_path):
    return pd.read_csv(file_path)

# Example usage: specify the path to your CSV file
file_path = Path("LTL 3.csv")  # Replace "your_file.csv" with your actual file name
df = load_dataframe(file_path)

# Now you can count the rows in the DataFrame
row_count = len(df)
null_values = df.isnull().sum().sum()
# print(f"The number of rows in the DataFrame is: {null_values}")

df = df.fillna(0)
null_values = df.isnull().sum().sum()
print(f"The number of rows in the DataFrame is: {null_values}")

# Convert CSV rows into LangChain documents
documents = []
for index, row in df.iterrows():
    content = " ".join([f"{col}: {row[col]}" for col in df.columns])  # Combine row data
    doc = Document(page_content=content, metadata={"row_index": index})
    documents.append(doc)

# Print the first document
print(len(documents))

llm = ChatOpenAI(temperature=0)

llm_transformer = LLMGraphTransformer(llm=llm)

graph_documents = llm_transformer.convert_to_graph_documents(documents)

stored_graph_documents = graph_documents


# Save the graph_documents to a pickle file
with open('graph_documents.pkl', 'wb') as f:
    pickle.dump(graph_documents, f)

print("Graph documents saved successfully.")

print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

