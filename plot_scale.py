import os
import matplotlib.pyplot as plt

MARKERS = ['s','o','d','p','^',',','.','v','<','>','1','2','3','4','8','*','P','h','H','+','x','X','D','|','_']
LINE_STYLES = ['-', '--', '-.', ':']
COLORS = ['#65a9d7', '#e9963e', '#f23b27', '#304f9e', '#449945']

def plot_scale(data_dict: dict, id_list: list=["0", "9", "89", "282", "1,836"],
               save_path: str="./resources/plot", manual_x: list=None, 
               file_name: str="scale", linewidth: float=1.0,
               additional_points: list=None, additional_index=None):
    # make the length of x,y 
    plt.figure(figsize=(6,6))

    # set the font of the scale 
    plt.tick_params(labelsize=13)
    
    # get global max,min value
    data_dict_temp = data_dict.copy()
    if additional_points is not None:
        data_dict_temp.update({"addition": additional_points})
    max_value = max([max(v.values()) for v in data_dict_temp.values()])
    min_value = min([min(v.values()) for v in data_dict_temp.values()])
    print("max_value: {}, min_value: {}".format(max_value, min_value))

    get_one_line = lambda task_id: [t.get(task_id,None) for t in data_dict.values()]
    normalize_values = lambda values: [v / max_value if v is not None else None for v in values]
    # normalize_values = lambda values: [(v-min_value) / (max_value-min_value) for v in values]
    map_to_percent = lambda values: [v * 100 if v is not None else None for v in values]
    get_values = lambda task_id: map_to_percent(normalize_values(get_one_line(task_id)))

    for i, task_id in enumerate(id_list):
        lb = "{} tasks".format(task_id) if int(task_id.replace(",", "")) != 0 else "0 task (vanilla)"
        # x axis index is normally distributed by default
        assert manual_x is None or len(manual_x) == len(data_dict.keys())
        x_index = list(data_dict.keys()) if manual_x is None else manual_x
        y_values = get_values(task_id)
        if None not in y_values:
            # all values are exist, plot directly
            plt.plot(x_index, 
                    y_values, 
                    color=COLORS[i%len(COLORS)], 
                    linestyle=LINE_STYLES[i%len(LINE_STYLES)], 
                    marker=MARKERS[i%len(MARKERS)], 
                    label=lb,
                    linewidth=linewidth)
    
    # plot additional points
    if additional_points is not None:
        for task_id,v in additional_points.items():
            lb = "{} tasks".format(task_id) if int(task_id.replace(",", "")) != 0 else "0 task (vanilla)"
            y = map_to_percent(normalize_values([v]))[0]
            plt.plot([additional_index],[y], marker=MARKERS[(i)%len(MARKERS)], label=lb)
            i += 1
        
    # set x axis normally distributed
    if manual_x is not None:
        plt.xticks(manual_x,list(data_dict.keys()))

    # set the font of xlabel and ylabel
    font = {'family' : 'arial',
    'weight' : 'normal',
    'size'   : 16,
    }
    plt.xlabel("Number of Model Parameters (Billions)",font)
    plt.ylabel("Normalized Performance (%)",font)


    legend = plt.legend(loc="upper left", frameon=True,shadow=True, fontsize='small') # x-large)

    # remove the margin around the pdf  
    plt.tight_layout()  
    plt.grid(axis = 'both', linestyle='-.', linewidth=0.5, color='gray', alpha=0.5)

    # save the pdf
    save_file = "{}.pdf".format(file_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, save_file)) 
    plt.show() 
    print("save the plot to {}\n".format(os.path.join(save_path, save_file)))

'''
==> customized data
'''
def plot_Flan_PaLM():
    Flan_PaLM = {
        "8": {"0":6.4, "9":8.3, "89":14.8, "282":20.5, "1,836":21.9},
        "62": {"0":28.4, "9":29.0, "89":33.4, "282":37.9, "1,836":38.8},
        "540": {"0":49.1, "9":52.6, "89":57.0, "282":57.5, "1,836":58.5},
    }
    
    plot_scale(data_dict=Flan_PaLM, 
               id_list=list(list(Flan_PaLM.values())[0].keys()), 
               file_name="Flan-PaLM",
               linewidth=1.5,
               manual_x=[0,0.5,1.5]
    )
    

def plot_Flan_T5():
    # Flan_T5 = {
    #     "0.08": {"8":24.67, "25":22.72, "50":23.48, "100":26.13, "200":28.03, "400":26.40, "800":25.71, "1,873":27.01},
    #     "0.25": {"8":22.35, "25":22.16, "50":22.81, "100":29.75, "200":28.85, "400":29.45, "800":28.44, "1,873":30.31},
    #     "0.78": {"8":23.63, "25":24.67, "50":26.90, "100":32.31, "200":37.78, "400":37.86, "800":37.71, "1,873":38.49},
    #     "3": {"8":29.33, "25":32.31, "50":36.33, "100":42.05, "200":41.97, "400":44.00, "800":45.95, "1,873":46.81},
    #     "11": {"8":43.64, "25":46.81, "50":52.42, "100":53.07, "200":54.51, "400":55.88, "800":53.84, "1,873":56.11},
    # }
    # plot_scale(data_dict=Flan_T5, id_list=["8", "25", "50", "100", "200", "400", "800", "1,873"], file_name="Flan-T5")
    
    Flan_T5 = {
        "0.08": {"8":24.67, "50":23.48, "200":28.03, "800":25.71, "1,873":27.01},
        "0.25": {"8":22.35, "50":22.81, "200":28.85, "800":28.44, "1,873":30.31},
        "0.78": {"8":23.63, "50":26.90, "200":37.78, "800":37.71, "1,873":38.49},
        "3": {"8":29.33, "50":36.33, "200":41.97, "800":45.95, "1,873":46.81},
        "11": {"8":43.64, "50":52.42, "200":54.51, "800":53.84, "1,873":56.11},
    }
    plot_scale(data_dict=Flan_T5, 
               id_list=list(list(Flan_T5.values())[0].keys()), 
               file_name="Flan-T5",
               linewidth=1.8,
               manual_x=[0.5,1,1.5,2,3]
    )
    
def plot_TkInstruct():
    TkInstruct = {
        "0.06": {"757":40.1},
        "0.22": {"757":42.1},
        "0.77": {"757":48.0},
        "3": {"757":54.0},
    }
    additional_points = {"8":43.7, "32":45.6, "128":48.4, "256":51.2, "512":52.9} 
    plot_scale(data_dict=TkInstruct, 
               id_list=list(list(TkInstruct.values())[-1].keys()), 
               file_name="TkInstruct",
               linewidth=2.0,
               additional_points=additional_points,
               additional_index="3",
               manual_x=[0.5,1,1.5,3]
    )


if __name__ == "__main__":
    plot_Flan_PaLM()
    plot_Flan_T5()
    # plot_TkInstruct()