import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import json
import gzip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)


def extend_doc(doc: str, num_return_sequences=3, max_length=64, do_sample=True, top_k=10):
    input_ids = tokenizer.encode(doc, return_tensors='pt',
                                 add_special_tokens=False)[0]

    start = 0
    window_size = 511
    stride = 211
    total_length = len(input_ids)
    flag = True
    chunks = []
    while flag:
        end = start + window_size
        if end >= total_length:
            flag = False
            end = total_length

        # end token
        input_chunk_ids = torch.concat([input_ids[start:end], torch.IntTensor([1])])

        # padding
        padding_size = abs(window_size + 1 - len(input_chunk_ids))
        input_chunk_ids = torch.concat(
            [input_chunk_ids, torch.IntTensor([0] * padding_size)])
        input_chunk_ids = torch.reshape(input_chunk_ids, [-1, 512])
        chunks.append(input_chunk_ids.to(device))
        start = start + stride

    extended_outputs = []
    for chunk in chunks:
        outputs = model.generate(
            input_ids=chunk,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences)

        seg_doc = tokenizer.decode(chunk[0], skip_special_tokens=True)
        for z in range(num_return_sequences):
            seg_doc += " " + tokenizer.decode(outputs[z], skip_special_tokens=True)
        extended_outputs.append(seg_doc)
    return extended_outputs


def test_extend_doc():
    text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the 
    industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it 
    to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, 
    remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem 
    Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem 
    Ipsum.Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the 
    industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it 
    to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, 
    remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem 
    Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem 
    Ipsum.""" * 10
    output = extend_doc(text)
    print(len(output))
    print(output)


if __name__ == "__main__":
    path = "$ZipFiles Folder$"
    zip_files = os.listdir(path)
    number_files = len(zip_files)
    for index, zipfile in enumerate(zip_files):
        with gzip.open(path + zipfile, 'rt', encoding='utf8') as f:
            lines = f.readlines()
            output_lines = ""
            for line in lines:
                doc_dict = json.loads(line)
                doc_text = doc_dict["body"].split()
                results = extend_doc(doc_text)
                doc_dict["body"] = results

                output_lines += json.dumps(doc_dict) + "\n"

        # writing the extended documents to a new file
        with open('extended_docs/' + zipfile.split('.')[0] + '.jsonl', 'w') as of:
            of.write(output_lines.strip())
        print(index + 1, zipfile, "is done!", number_files - index - 1, "left!")
