import torch,model_ASR,model_Translate,model_TTS,configs,time

if torch.cuda.is_available():
    devices="cuda:0"


# Start time to control time
start_time=time.time() 



# Asr Model
Result_text_ASR=model_ASR.ASR_Model(devices=devices,
                                    model_type=configs.MODEL_TYPE,
                                    audio_path=configs.Audio_Path_ASR,
                                    save_path=configs.Save_Path_ASR,
                                    SAVE_ASR=configs.SAVE_ASR).forward()



# Translate Model
Result_text_Translated=model_Translate.Translate_Model(devices=devices,
                                                       Text=Result_text_ASR,
                                                       SOURCE_TEXT_LANG=configs.SOURCE_TEXT_LANG,
                                                       TRANSLATED_LAN=configs.TRANSLATED_LAN,
                                                       SAVE_PATH_TRANSLATE=configs.SAVE_PATH_TRANSLATE).forward()

# TTS Model Tacotron
Result_Audios=model_TTS.TEXT_TO_SPEECH(devices=devices,
                                       Text=Result_text_Translated,
                                       VISUALIZE_WAVE_AND_SPEC=configs.VISUALIZE_WAVE_AND_SPEC,
                                       SAVE_AUDIO_PATH=configs.SAVE_AUDIO_PATH).forward()



# Finish time to control time
finish_time=time.time()


print(f"\n***Training Has Taken {((finish_time-start_time)/60):.3f} Minute***")