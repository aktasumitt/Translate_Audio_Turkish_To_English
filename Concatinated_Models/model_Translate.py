from transformers import MBartForConditionalGeneration,MBart50TokenizerFast



class Translate_Model():
   
    def __init__(self,devices,Text:list,SOURCE_TEXT_LANG:str,TRANSLATED_LAN:str,SAVE_PATH_TRANSLATE:str):
        
        
        self.devices=devices                          # CPU or CUDA
        self.Text=Text                                # TEXT FROM ASR_Model output
        self.SOURCE_TEXT_LANG=SOURCE_TEXT_LANG        # Inıtıal Lenguage That You Want to Translate
        self.TRANSLATED_LAN=TRANSLATED_LAN            # Language that You Want To reach Leanguage with translate
        self.SAVE_PATH_TRANSLATE=SAVE_PATH_TRANSLATE  # Path that You want to save translated text (.txt format)

    
    def Create_Model_Tokenizer(self):
    
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        return model,tokenizer

    
    
    def forward(self):    

        print("Translating is starting....")
  
  
        # Create Model and Tokenizer
        Model_Bart,Tokenizer=self.Create_Model_Tokenizer()
        Model_Bart.to(self.devices)


        # Create your Source language
        Tokenizer.src_lang = self.SOURCE_TEXT_LANG


        # Tokenize your source sentences
        tokenized_text = Tokenizer(self.Text, return_tensors="pt")
        tokenized_text.to(self.devices)


        # Generate translated predictions with Bart Model
        output_translate=Model_Bart.generate(**tokenized_text,forced_bos_token_id=Tokenizer.lang_code_to_id[self.TRANSLATED_LAN])


        # Tranforming from tokens to text
        out_text=Tokenizer.batch_decode(output_translate, skip_special_tokens=True)

        with open(self.SAVE_PATH_TRANSLATE,"w") as dosya:
            dosya.write(f"Input Text:\n{self.Text}\n\nTranslated Text:\n{out_text}")


        # Print Generated TEXT
        print("Generated Text: \n",out_text)
        
        return out_text