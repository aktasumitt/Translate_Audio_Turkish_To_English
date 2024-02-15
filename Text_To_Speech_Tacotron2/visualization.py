import matplotlib.pyplot as plt

def plot_spectogram(spec_list):  
    
    for i in range(len(spec_list)):
        plt.subplot(1,len(spec_list),i+1)
        plt.imshow(spec_list[i][0].cpu().detach(), origin="lower", aspect="auto")
        plt.show()


def Plot_Waveform_and_Spec(waveforms_list, spec_list):
    
    for i,waveforms in enumerate(waveforms_list,0):
        
        waveforms = waveforms.cpu().detach()

        fig, [ax1, ax2] = plt.subplots(2, 1)
        ax1.plot(waveforms[0])
        ax1.set_xlim(0, waveforms.size(-1))
        ax1.grid(True)
        ax2.imshow(spec_list[i][0].cpu().detach(), origin="lower", aspect="auto")
        plt.show()