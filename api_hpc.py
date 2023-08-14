import gradio as gr
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings #Embedding
from langchain.vectorstores import Chroma #Vector Space
from googletrans import Translator
translator = Translator()

model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    cache_dir='tmp',
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    cache_dir='tmp',
    use_auth_token=hf_auth    
)

stop_list = ['\nUser:', '\n```\n','system', ':',""]

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('luat', glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, #độ dài của chunk
    chunk_overlap = 0, #độ trùng lặp của các chunk
    length_function = len, #cách tính độ dài của chunk
    add_start_index = True, #có thể start_index vào đầu các chunk không
    )
all_splits = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    cache_folder='tmp',
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore3 = Chroma.from_documents(documents=all_splits, embedding=hf)
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore3.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

def QA_trans(query):
    query_english =translator.translate(query,src="vi", dest="en").text
    answer_eng = qa_chain({"query": query_english})['result']
    answer_vi = translator.translate(answer_eng,src="en", dest="vi").text
    return answer_vi

examples = [
    ["Tôi đi xe ô tô, lỗi đi sai làn đường sẽ bị phạt bao nhiêu tiền"],
    ["Đi xe máy không đội mũ bảo hiểm phạt bao nhiêu"],
    ["Giải thích cho tôi điều 12 của Luật lao động"],
]

demo = gr.Interface(
    fn=QA_trans,
    inputs=gr.inputs.Textbox(lines=5, label="Đầu vào"),
    outputs=gr.outputs.Textbox(label="Câu trả lời"),
    examples=examples
)

demo.launch(
    server_port=5556
)