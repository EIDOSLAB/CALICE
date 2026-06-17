import torch
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import itertools

palette = sns.color_palette("tab10")



def plot_rate_distortion_perception(json_data, epoch = 0., eest = 'rate-distortion-perception', log_wandb = False, save_fig = False, file_name = 'out.pdf' ):
    plt.figure(figsize=(12, 5))

    # Genera colori per ogni categoria dinamicamente
    category_colors = {}
    color_cycle = itertools.cycle(["blue", "red", "green", "purple", "orange", "brown", "pink", "cyan"])

    # Associa un colore a ciascuna categoria trovata
    for category in json_data.keys():
        category_colors[category] = next(color_cycle)

    # *** Grafico Rate-Distortion (BPP vs PSNR) ***
    plt.subplot(1, 2, 1)
    for category, data in json_data.items():
        color = category_colors[category]
        for beta, values in data.items():
            label = f"{category}"
            beta_value = beta.split("_")[1]
            plt.plot(values["bpp"], values["PSNR"], label=label, color=color, linewidth=0.8)

            # Etichetta sopra la curva
            mid_idx = len(values["bpp"]) // 2  # Punto centrale per la scritta
            plt.text(values["bpp"][mid_idx], values["PSNR"][mid_idx], f"β = {beta_value}", fontsize=10, color='black')

    plt.xlabel("BPP")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Curve")
    plt.legend()
    plt.grid()

    # *** Grafico Rate-Perception (BPP vs LPIPS) ***
    plt.subplot(1, 2, 2)
    for category, data in json_data.items():
        color = category_colors[category]
        for beta, values in data.items():
            label = f"{category}"
            beta_value = beta.split("_")[1]
            plt.plot(values["bpp"], values["LPIPS"], label=label, color=color, linewidth=0.8)

            # Etichetta sopra la curva
            mid_idx = len(values["bpp"]) // 2
            plt.text(values["bpp"][mid_idx], values["LPIPS"][mid_idx], f"β = {beta_value}", fontsize=10, color='black')

    plt.xlabel("BPP")
    plt.ylabel("LPIPS (Lower is Better)")
    plt.title("Rate-Perception Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if log_wandb:
        wandb.log({f"{eest}/epoch":epoch,
                f"{eest}/rate distorsion trade-off": wandb.Image(plt)}, step=epoch)      

    if save_fig:
        plt.savefig(file_name) 
    plt.close()  

def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression",metric = 'PSNR', save_fig = False, file_name = None, log_wandb = True, is_psnr = True):

    chiavi_da_mettere = list(psnr_res.keys())


    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[i],'-']
        legenda[c]["legends"] = c
        legenda[c]["symbols"] = ["*"]*300
        legenda[c]["markersize"] = [5]*300    



    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))


    list_names = list(psnr_res.keys())

    if is_psnr:
        minimo_bpp, minimo_psnr = 10000,1000
        massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        #symbols = legenda[type_name]["symbols"]
        #markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]


        bpp = torch.tensor(bpp).cpu()
        psnr = torch.tensor(psnr).cpu()
    
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=8)
        
        plt.plot(bpp, psnr, marker="o", markersize=4, color =  colore)

        if is_psnr:
            for j in range(len(bpp)):
                if bpp[j] < minimo_bpp:
                    minimo_bpp = bpp[j]
                if bpp[j] > massimo_bpp:
                    massimo_bpp = bpp[j]
                
                if psnr[j] < minimo_psnr:
                    minimo_psnr = psnr[j]
                if psnr[j] > massimo_psnr:
                    massimo_psnr = psnr[j]


    if is_psnr:
        minimo_psnr = int(minimo_psnr)
        massimo_psnr = int(massimo_psnr)
        psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
        plt.yticks(psnr_tick)
    
    plt.ylabel(metric, fontsize = 30)   
    


    #print(minimo_bpp,"  ",massimo_bpp)

    if is_psnr:
        bpp_tick = [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
        plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='best', fontsize = 25)



    plt.grid(True)
    if log_wandb:
        wandb.log({f"{eest}":epoch,
                f"{eest}/rate distorsion trade-off": wandb.Image(plt)}, step=epoch)      

    if save_fig:
        plt.savefig(file_name) 
    plt.close()  