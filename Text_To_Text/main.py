import torch
import model,config

# Control and use GPU
if torch.cuda.is_available():
    devices="cuda"


# Create Model and Tokenizer
Model_Bart,Tokenizer=model.Create_Model_Tokenizer()
Model_Bart.to(devices)


# Create your Source language
Tokenizer.src_lang = config.SOURCE_TEXT_LANG


# Tokenize your source sentences
tokenized_text = Tokenizer(config.TEXT, return_tensors="pt")
tokenized_text.to(devices)


# Generate translated predictions with Bart Model
output_translate=Model_Bart.generate(**tokenized_text,forced_bos_token_id=Tokenizer.lang_code_to_id[config.TRANSLATED_LAN])


# Tranforming from tokens to text
out_text=Tokenizer.batch_decode(output_translate, skip_special_tokens=True)

with open(config.SAVE_PATH,"w") as dosya:
    dosya.write(f"Input Text:\n{config.TEXT}\n\nTranslated Text:\n{out_text}")


# Print Generated TEXT
print("Generated Text: \n",out_text)

