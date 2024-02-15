import torch,model,config

# Control and use GPU
if torch.cuda.is_available():
    devices="cuda"


# Create Model
# You can use model_type with ("tiny","base","small","medium","large")
whisper_model=model.Create_Model(devices=devices,model_type="medium")



# Predict transcribe from audio
result=model.Predicting(model=whisper_model,audio_path=config.Audio_Path)

print("Generated Text: ",result["text"])


# Save Predicted text
if config.SAVE==True:
    model.Save_predicted(result=result,save_path=config.Save_Path)
