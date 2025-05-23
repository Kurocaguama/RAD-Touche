from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from check_polarity import polarity
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import torch, json, re

# Argument Retrieval Info
elastic_url = 'https://touche25-rad.webis.de/arguments/'
es_client = Elasticsearch(elastic_url, retry_on_timeout=True)
index_name = 'claimrev' 

# Sentiment Analysis Info
pysent_analyzer = create_analyzer(task = 'sentiment', lang = 'en')
vadsent_analyzer = SentimentIntensityAnalyzer()


def py_sen_values(output):
    """
    Exctracts pysentimiento class probabilities.

    Extraction order: NEG-NEU-POS
    """
    aux = output.probas
    a = round(aux['NEG'], 2)
    b = round(aux['NEU'], 2)
    c = round(aux['POS'], 2)
    return a, b, c

def vad_sen_values(output):
    return output['neg'], output['neu'], output['pos']

def polarity(texto):
    """
    Returns one of three values: (-1, 0, 1) based on the user's claim's polarity. Based on this the LLM's behavior is modified.
    """
    pysent = pysent_analyzer.predict(texto)
    vadsent = vadsent_analyzer.polarity_scores(texto)
    
    py_neg, py_neu, py_pos = py_sen_values(pysent)
    vs_neg, vs_neu, vs_pos = vad_sen_values(vadsent)
    neg = (py_neg + vs_neg)/2
    neu = (py_neu + vs_neu)/2
    pos = (py_pos + vs_pos)/2
    aux = max(neg, neu, pos)
    if aux == neg:
        return -1
    elif aux == neu:  
        return 0
    else:
        return 1
    

class LLM_Agent():
    """
    Agent class. Contains all info and functions for argument analysis and debate generation.
    """
    def __init__(
            self,
            model_id
    ):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype = torch.bfloat16)
        self.gen_config = GenerationConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.dev)
        self.prompt = 'Null' # Base case is non-existent.
        
        # Database interaction
        self.embedding_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
        self.sup_field = 'supports_embedding_stella'
        self.attack_field = 'attacks_embedding_stella'


    def select_prompt(self, input, arguments):
        pol_val = polarity(input)
        if pol_val == 0:
            self.prompt = """
                    You are a debate agent. Refute the user's claim based on the following arguments.
                    USER CLAIM: {}.
                    ARGUMENTS: {}.
                """.format(input, arguments)
        elif pol_val == 1:
            self.prompt = """
                    You are a debate agent. The user's claim is positive, refute it with a negative argument from the follwing.
                    USER CLAIM: {}.
                    ARGUMENTS: {}.
                """.format(input, arguments)
        else:
            self.prompt = """
                    You are a debate agent. Based on your judgement of the user's claim refute it with one of the following arguments.
                    USER CLAIM: {}.
                    ARGUMENTS: {}.
                """.format(input, arguments)
            

    def query_search(self, input):
        """
        *agregar cockmentarios de Diego*
        """
        answer = []
        query_embedding = self.embedding_model.encode(input, prompt_name='s2p_query')

        sup_response = es_client.search(
            index = index_name,
            knn = {
                'field': self.sup_field,
                'query_vector': query_embedding,
                'k': 10,
                'num_candidates': 100
            },
        )

        for hit in sup_response['hits']['hits']:
            del hit['_source']['attacks_embedding_stella']
            del hit['_source']['supports_embedding_stella']
            del hit['_source']['text_embedding_stella']

            json_string = json.dumps(hit['_source'], indent=2)
            answer.append(json_string)

        attack_response = es_client.search(
            index = index_name,
            knn = {
                'field': self.attack_field,
                'query_vector': query_embedding,
                'k': 10,
                'num_candidates': 100
            },
        )

        for hit in attack_response['hits']['hits']:
            del hit['_source']['attacks_embedding_stella']
            del hit['_source']['supports_embedding_stella']
            del hit['_source']['text_embedding_stella']

            json_string = json.dumps(hit['_source'], indent=2)
            answer.append(json_string)
        
        return answer
    

    def get_pos_neg(self, string):
        """
        Filtering of text cases.
        """
        attack_start = re.search('\"attacks\"', string).end() + 2
        support_start = re.search('\"supports\"', string).end() + 2
        text_start = re.search('\"text\"', string).end()
        attack = string[attack_start:support_start-16] # Character level adjustment
        support = string[support_start: text_start-10]
        return attack, support


    def debate(self):
        """
        Full debate pipeline, returns a single answer.
        """
        debate_input = input('User input: ')
        args = self.query_search(debate_input) 
        attack, support = self.get_pos_neg(args)
        both_args = attack + support

        # Hasta aquí funciona.
        self.select_prompt(debate_input, both_args)
        inputs = self.tokenizer(self.prompt, return_tensors = 'pt').to(self.dev)
        outputs = self.model.generate(**inputs, max_new_tokens = 100)
        answer = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        #print(answer[len(self.prompt):])
        return answer[len(self.prompt):]
    

hf_key = None # For our experiments we used HuggingFace-hosted LLMs. In order for this to work you need to have a HuggingFace key.
login(hf_key)

# Supports Llama 3 models.
model_id = 'meta-llama/Llama-3.2-1B' # Can be modified for a better performing model.

chat_agent = LLM_Agent(model_id)
chat_agent.debate() # Just insert an initial argument and the answer will be given.

#aux = chat_agent.query_search('I believe that school uniforms are good for the children')
#print(chat_agent.get_pos_neg(aux[0]))