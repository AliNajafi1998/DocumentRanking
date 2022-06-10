import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)


def extend_doc(doc: str, num_return_sequences=3, max_length=64, do_sample=True, top_k=10):
    input_ids = tokenizer.encode(doc, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences)

    for i in range(num_return_sequences):
        doc += " " + tokenizer.decode(outputs[i], skip_special_tokens=True)
    return doc
