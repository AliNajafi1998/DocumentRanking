import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import json
import gzip
import more_itertools

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


if __name__ == "__main__":
    path = "/DATA/users/alinajafi/documentRanking/msmarco_v2_doc/"
    zip_files = os.listdir(path)
    number_files = len(zip_files)
    for index, zipfile in enumerate(zip_files):
        with gzip.open(path + zipfile, 'rt', encoding='utf8') as f:
            lines = f.readlines()
            output_lines = ""
            for line in lines:
                doc_dict = json.loads(line)
                doc_text = doc_dict["body"].split()

                # need to split it and get the queries then concat
                sub_docs = list(more_itertools.windowed(doc_text, n=512, step=510))
                sub_docs = [" ".join(sub_doc).strip() for sub_doc in sub_docs]
                results = [extend_doc(doc) for doc in sub_docs]
                doc_dict["body"] = results

                output_lines += json.dumps(doc_dict) + "\n"

        # writing the extended documents to a new file
        with open(zipfile.split('.')[0], 'w') as of:
            of.write(output_lines.strip())
        print(index + 1, zipfile, "is done!", number_files - index - 1, "left!")
