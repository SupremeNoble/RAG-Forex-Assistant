# llm.py
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from retriever import retriever

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Choose a HF-hosted Llama 3 model
model_name = "meta-llama/Llama-3-3b-hf"

# Load tokenizer and model from HF
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text-generation pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2
)

# Wrap in LangChain HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # concat retrieved docs
    retriever=retriever,
    return_source_documents=True
)