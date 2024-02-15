import torch,time
from scipy.io.wavfile import write
from IPython.display import Audio
import tqdm


def Tacotron_processor(text_list,devices,processor,tacotron2):     
    
    spec_list=[]
    spec_lengths_list=[]
    print("\nTacotron is predicting texts...")
    
    progrss_bar=tqdm.tqdm(range(len(text_list)),"Training_progress",position=0,leave=True)
    
    for text in text_list:    
        with torch.inference_mode():
            processed, lengths = processor(text)
            processed = processed.to(devices)
            lengths = lengths.to(devices)
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
            spec_lengths_list.append(spec_lengths)
            spec_list.append(spec)
            
            progrss_bar.update(1)
    progrss_bar.close()
    print("\nPredicting is finished...")
    return spec_list,spec_lengths_list



def WaveGlow_Vocoder(spec_list,spec_lengths_list,vocoder):
    waveform_list=[]
    lengths_list=[]
    
    print("\nWaveGlow is predicting texts...")
    progrss_bar=tqdm.tqdm(range(len(spec_list)),"Training_progress",position=0,leave=True)

    
    for i in range(len(spec_list)):
        waveforms, lengths = vocoder(spec_list[i], spec_lengths_list[i])
        waveform_list.append(waveforms)
        lengths_list.append(lengths)
       
        progrss_bar.update(1)
    progrss_bar.close()
    print("\nPredicting is finished")
    return waveform_list,lengths_list



def Saving_Audio(waveforms_list,sample_rate,saving_path):
        
    concat_waveform = waveforms_list[0]

    for waveform in waveforms_list[1:]:
        concat_waveform = torch.cat((concat_waveform, waveform), dim=1)

    concat_waveform = concat_waveform.cpu().numpy()   
    write((f"{saving_path}"+f"\\test.wav"), sample_rate, concat_waveform[0])  
    
    print("\nSaving is finished")
