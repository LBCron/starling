import sys,time,json,resource
import numpy as np, torch, onnxruntime as ort
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
S=sys.argv[1]
torch.set_num_threads(4)
raw=Tokenizer.from_file(S+"/tokenizer.json")
print("declared trunc",raw.truncation,"pad",raw.padding)
long_text=" ".join(f"token{i} the quick brown fox" for i in range(4000))
e=raw.encode("q",long_text);print("raw file encode len",len(e.ids),"mask sum",sum(e.attention_mask))
e=raw.encode("q","d");print("raw file short pair len",len(e.ids),"mask sum",sum(e.attention_mask))
at=AutoTokenizer.from_pretrained(S)
print(type(at),"model_max_length",at.model_max_length)
print("auto truncation=True",len(at("q",long_text,truncation=True)["input_ids"]))
print("auto no truncation",len(at("q",long_text)["input_ids"]))
print("auto max_length=512",len(at("q",long_text,truncation=True,max_length=512)["input_ids"]))
print("type ids key",list(at("q","d").keys()))
r=at("what is the capital of China?","Beijing");print(r["input_ids"])
raw.no_padding();raw.enable_truncation(8192);print(raw.encode("what is the capital of China?","Beijing").ids)
model=AutoModelForSequenceClassification.from_pretrained(S,dtype=torch.float32).eval()
print("attn",model.config._attn_implementation)
meager=AutoModelForSequenceClassification.from_pretrained(S,dtype=torch.float32,attn_implementation="eager").eval()
o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
s=ort.InferenceSession(S+"/onnx/model.onnx",o,providers=["CPUExecutionProvider"])
raw.no_truncation()
ids=raw.encode("the quick brown fox jumps over the lazy dog "*1000,"how to implement quick sort in python, and other things "*1000).ids
print("len",len(ids))
def pair(n):
    return ids[:n-1]+[50282]
for n in [8,64,127,129,130,200,257,512,1024,2048,4096,8000]:
    x=np.array([pair(n)],dtype=np.int64);mk=np.ones_like(x)
    t0=time.time();
    with torch.inference_mode(): a=model(input_ids=torch.from_numpy(x),attention_mask=torch.from_numpy(mk)).logits.numpy()
    t1=time.time();b=s.run(["logits"],{"input_ids":x,"attention_mask":mk})[0];t2=time.time()
    if n<=2048:
        with torch.inference_mode(): c=meager(input_ids=torch.from_numpy(x),attention_mask=torch.from_numpy(mk)).logits.numpy()
    else: c=a
    print(n,"pt",a.ravel(),"ort",b.ravel(),"shape",b.shape,"d",abs(a-b).max(),"eager d",abs(a-c).max(),f"pt {t1-t0:.1f}s ort {t2-t1:.1f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}G",flush=True)
rows=[pair(n) for n in (8,64,200,600)]
w=600
for padid in (0,50283):
    x=np.array([r+[padid]*(w-len(r)) for r in rows],dtype=np.int64);mk=np.array([[1]*len(r)+[0]*(w-len(r)) for r in rows],dtype=np.int64)
    with torch.inference_mode(): a=model(input_ids=torch.from_numpy(x),attention_mask=torch.from_numpy(mk)).logits.numpy()
    with torch.inference_mode(): ae=meager(input_ids=torch.from_numpy(x),attention_mask=torch.from_numpy(mk)).logits.numpy()
    b=s.run(["logits"],{"input_ids":x,"attention_mask":mk})[0]
    singles=[]
    singlesort=[]
    for r in rows:
        y=np.array([r],dtype=np.int64)
        singlesort.append(s.run(["logits"],{"input_ids":y,"attention_mask":np.ones_like(y)})[0].item())
        with torch.inference_mode(): singles.append(model(input_ids=torch.from_numpy(y),attention_mask=torch.ones_like(torch.from_numpy(y))).logits.item())
    print("pad",padid,"pt batch",a.ravel(),"eager batch",ae.ravel(),"ort batch",b.ravel(),"pt single",singles,"ort single",singlesort,flush=True)
