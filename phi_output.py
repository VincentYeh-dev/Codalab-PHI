import glob
import sys
import os
import json
import time

import torch
from transformers import GPTJForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from normalization import clear_file, read_file2text, gen_text_normalization

TOKEN_BOS = '<|endoftext|>'
TOKEN_EOS = '<|END|>'
TOKEN_PAD = '<|pad|>'
TOKEN_SEP = '\n\n####\n\n'


def createTokenizer():
    # special_tokens_dict: List[str] = [bos, eos, pad, sep]
    special_tokens_dict = {'eos_token': TOKEN_EOS, 'bos_token': TOKEN_BOS, 'pad_token': TOKEN_PAD,
                           'sep_token': TOKEN_SEP}

    tokenizer = AutoTokenizer.from_pretrained(plm, revision="step3000")
    tokenizer.padding_side = 'left'

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    return tokenizer


def createModel(tokenizer):
    config = AutoConfig.from_pretrained(plm,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        sep_token_id=tokenizer.sep_token_id,
                                        output_hidden_states=False)
    model = AutoModelForCausalLM.from_pretrained(plm, revision="step3000", config=config)
    model.resize_token_embeddings(len(tokenizer))
    return model


plm = "EleutherAI/pythia-70m-deduped"
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch.device is", device)
    args = sys.argv
    print(args)
    # model_file_path = 'models/pythia/70m_epoch_30_batch_2_lr5e_5.pt'
    # folder_path = 'data/codalab/1st/validation'
    # results_path = 'answer_70m_epoch_30_batch_2_lr5e_5.txt'
    # gen_text_path = 'gen_text_70m_epoch_30_batch_2_lr5e_5.json'

    # plm = args[1]
    model_file_path = args[1]
    glob_path = args[2]
    results_path = args[3]
    gen_text_path = "gen.json"

    clear_file(results_path)
    # clear_file(gen_text_path)

    tokenizer = createTokenizer()
    model = createModel(tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_file_path))
    model.to(device)

    files = glob.glob(os.path.join(glob_path))
    print("找到{}個檔案".format(len(files)))

    gen_text_dict = {}

    start = time.time()
    for file_path in files:
        results_list = []
        file_text_dict = {}
        medical_report = read_file2text(path=file_path)
        fileName = os.path.splitext(os.path.basename(file_path))[0]
        # print(fileName)
        with open(file_path) as f:
            texts = f.readlines()

        # article = "".join(texts)
        # prompt_template = f"{article}"

        for index, text in enumerate(texts):
            input_ids = tokenizer.encode(
                text,
                return_tensors="pt",
                truncation=True, max_length=1000).to(device)
            generated_tokens_with_prompt = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=100
            ).to(device)
            generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt)

            # generated_text_answer = generated_text_with_prompt[0][len(text):]

            sep_index = generated_text_with_prompt[0].find(TOKEN_SEP)
            eos_index = generated_text_with_prompt[0].rfind(TOKEN_EOS)

            if sep_index == -1:
                continue
            if eos_index == -1:
                continue

            start_index = sep_index + len(TOKEN_SEP)

            # prompt_field = generated_text_with_prompt[0][0:sep_index]
            answer_field = generated_text_with_prompt[0][start_index:eos_index]

            print(answer_field)
            file_text_dict[index] = answer_field

            result = gen_text_normalization(fileName, answer_field, medical_report, index + 1)
            if result is not None:
                results_list.append(result)
                print(result)
        # break
        # save to results_path
        with open(results_path, 'a', encoding='utf-8') as file:
            for result in results_list:
                file.write(str(result) + '\n')

        # save to gen_text_path
        gen_text_dict[str(fileName)] = file_text_dict
        with open(gen_text_path, "w", encoding='utf-8') as json_file:
            json.dump(gen_text_dict, json_file, indent=4)

        print("\033[93m------ Saved ------\033[0m")
    print("\033[92m------ Done ------\033[0m")
    end = time.time()

    print(f"time:{int(end-start)}s")
