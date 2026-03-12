import os
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

print("\n=== INICIALIZANDO ARGUS ===")

# === Conexão com Pinecone
os.environ["PINECONE_API_KEY"] = "pcsk_7Amgef_RcM74ZD1P8v7sQM78WNsXtomMX2v9VcS5W8LzMH2TAgcYdfoXNMqudAJgqqqRjd"
# INDEX_NAME = "argus-index-recursivefirst"
INDEX_NAME = "argus-index-recursivesecond"

print("[1] Carregando modelos de Embedding e Cross-Encoder...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model_rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("[2] Conectando aos vetores existentes no Pinecone...")
# ATENÇÃO AQUI: from_existing_index apenas LÊ os dados, não duplica!
vs_macro = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace="macro")
vs_micro = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace="micro")

# == Função retrieval duplo
def retrieve(query, top_k_macro=10, final_k=5):
    print(f"\n=== INICIANDO RETRIEVAL PARA: '{query}' ===")
    
    # FASE 1: MACRO
    print(f"\n[Fase 1: MACRO] Buscando os {top_k_macro} chunks mais próximos no Pinecone...")
    res_macro = vs_macro.similarity_search(query, k=top_k_macro)
    
    print(f"[Fase 1: MACRO] Aplicando Rerank nos {len(res_macro)} chunks encontrados...")
    pairs_macro = [[query, d.page_content] for d in res_macro]
    scores_macro = model_rerank.predict(pairs_macro)
    top_idx_macro = np.argsort(scores_macro)[::-1][:5]
    
    print(f"\n=== TOP 5 TEXTOS DA FASE 1 (MACRO) ===")
    for i, idx in enumerate(top_idx_macro, 1):
        doc = res_macro[idx]
        score = scores_macro[idx]
        print(f" {i}. [Doc ID: {doc.metadata['doc_id']}] | Score: {score:.4f} | Texto: {doc.page_content[:60]}...")
    
    # Extração dos IDs únicos
    doc_ids = list(set([res_macro[i].metadata["doc_id"] for i in top_idx_macro]))
    print(f"\n[Fase 1: MACRO] Documentos originais vencedores (IDs): {doc_ids}")
    
    # FASE 2: MICRO
    print(f"\n[Fase 2: MICRO] Buscando chunks MENORES restritos aos documentos {doc_ids}...")
    res_micro = vs_micro.similarity_search(query, k=top_k_macro, filter={"doc_id": {"$in": doc_ids}})
    
    print(f"[Fase 2: MICRO] Aplicando Rerank final nos {len(res_micro)} chunks filtrados...")
    pairs_micro = [[query, d.page_content] for d in res_micro]
    scores_micro = model_rerank.predict(pairs_micro)
    top_idx_micro = np.argsort(scores_micro)[::-1][:final_k]
    
    final_docs = [res_micro[i].page_content for i in top_idx_micro]
    
    print(f"\n=== TOP {final_k} TEXTOS FINAIS SELECIONADOS (MICRO) ===")
    for i, idx in enumerate(top_idx_micro, 1):
        doc = res_micro[idx]
        score = scores_micro[idx]
        print(f" {i}. [Doc ID: {doc.metadata['doc_id']}] | Score: {score:.4f} | Texto: {doc.page_content[:60]}...")
    print("==================================================")
    
    return "\n".join(final_docs)

# ==========================================
# 3. CARREGAMENTO DO LLM E GERAÇÃO
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Força o dispositivo para CPU se não houver GPU (comum em deploy)
device = torch.device("cpu") 

print(f"\n[3] Carregando modelo LLM '{model_name}' em modo CPU...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_gen = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float32,  # CPU trabalha melhor com float32
    device_map={"": "cpu"}      # Força explicitamente o mapeamento para CPU
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def rag_answer(query, max_tokens=150):
    context = retrieve(query)
    
    print("\n[4] Contexto recuperado. Gerando resposta com o LLM...")
    prompt = f"<|im_start|>system\nYou are Argus, an oracle of Greek mythology. Provide a direct, factual, and complete answer using only the provided context. Do not use unnecessary filler words.<|im_end|>\n<|im_start|>user\nContext: {context}\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
    # prompt = f"<|im_start|>system\nYou are Argus, an oracle of Greek mythology. Provide a comprehensive, and engaging answer using only the provided context. If the context has multiple facts, connect them naturally into a cohesive and single paragraph.<|im_end|>\n<|im_start|>user\nContext: {context}\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    with torch.no_grad():
        outputs = model_gen.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            eos_token_id=stop_token_id if stop_token_id is not None else tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.split("<|im_start|>")[0].strip()

# if __name__ == "__main__":
#     print("\n--- TESTE DE QUERY ---")
#     query = "Tell me about Orpheus and Eurydice."
#     resposta = rag_answer(query)
#     print(f"\n[RESULTADO FINAL]:\n{resposta}")