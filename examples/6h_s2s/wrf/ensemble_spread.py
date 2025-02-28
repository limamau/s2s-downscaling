import os, tomllib
import matplotlib.pyplot as plt
import numpy as np

from data.surface_data import ForecastEnsembleSurfaceData

EVENT_LENGTH = 8


def run_spread_analysis(ens_s2s, ordered_idxs_choice):
    # partition of the data between the two events
    num_event_idxs = EVENT_LENGTH
    precip_2018 = ens_s2s.precip[:,:,:num_event_idxs]
    precip_2021 = ens_s2s.precip[:,:,num_event_idxs:]
    
    # average precipitation per ensemble member and lead time
    precip_2018_mean = np.mean(precip_2018, axis=(2, 3, 4))
    precip_2021_mean = np.mean(precip_2021, axis=(2, 3, 4))
    
    # crate figs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, "figs", "ensemble_spread")
    os.makedirs(figs_dir, exist_ok=True)
    
    # plot of the ensemble member by the mean precipitatation
    # for each year and lead time
    for year, precip_mean in zip(("2018", "2021"), (precip_2018_mean, precip_2021_mean)):
        for lead_time_idx in range(precip_mean.shape[0]):
            plt.plot(
                precip_mean[lead_time_idx],
                label=str(year)+", "+ens_s2s.lead_time[lead_time_idx],
            )
        plt.legend()
        plt.savefig(os.path.join(figs_dir, f"spread_{year}.png"))
        plt.close()
        
    # one ensemble member indice equally spaced in the ordered ensemble members
    # per mean precipitation for each year and lead time
    choices_dict = {
        "2018": {
            "1-week": [], # list to be populate with #num_wrf_ensembles indices
            "2-week": [],
            "3-week": [],
        },
        "2021": {
            "1-week": [],
            "2-week": [],
            "3-week": [],
        },
    }
    for year, precip_mean in zip(("2018", "2021"), (precip_2018_mean, precip_2021_mean)):
        for lead_time_idx, lead_time_name in enumerate(ens_s2s.lead_time):
            order = np.argsort(precip_mean[lead_time_idx])
            for i in ordered_idxs_choice:
                choices_dict[year][lead_time_name].append(order[i])
            
    # plot of the ordered ensemble members by the mean precipitation
    # for each year and lead time
    text_y_ref = np.mean(ens_s2s.precip) * 2
    for year, precip_mean in zip(("2018", "2021"), (precip_2018_mean, precip_2021_mean)):
        for lead_time_idx, lead_time_name in enumerate(ens_s2s.lead_time):
            order = np.argsort(precip_mean[lead_time_idx])
            plt.plot(
                precip_mean[lead_time_idx, order],
                label=str(year)+", "+lead_time_name,
            )
            for enum, choice in enumerate(choices_dict[year][lead_time_name]):
                plt.axvline(np.where(order == choice)[0], color="black", linestyle="--")
                plt.text(np.where(order == choice)[0], text_y_ref, enum, color="black")
        plt.xlabel("Ordered ensemble members")
        plt.ylabel("Mean precipitation")
        plt.legend()
        plt.savefig(os.path.join(figs_dir, f"ordered_spread_{year}.png"))
        plt.close()
        
    # print choices
    print("choices:")
    for year in choices_dict.keys():
        print("  year:", year)
        for lead_time_name in choices_dict[year].keys():
            print("   lead time:", lead_time_name)
            print(choices_dict[year][lead_time_name])
        print()
        
    # print mean precipitation of each choice
    print("mean precipitation of each choice:")
    for year, precip_mean in zip(("2018", "2021"), (precip_2018_mean, precip_2021_mean)):
        for lead_time_idx, lead_time_name in enumerate(ens_s2s.lead_time):
            print("year:", year, "lead time:", lead_time_name)
            for choice in choices_dict[year][lead_time_name]:
                print(np.mean(precip_mean[lead_time_idx, choice]))
            print()


def main():
    # directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "../dirs.toml"), "rb") as f:
        dirs = tomllib.load(f)
    base = dirs["main"]["base"]
    test_data_dir = os.path.join(base, dirs["subs"]["test"])
    
    # extra configurations
    ens_s2s_file = os.path.join(test_data_dir, "ens_s2s_nearest_low-pass.h5")
    ens_s2s = ForecastEnsembleSurfaceData.load_from_h5(ens_s2s_file, ["precip"])
    ordered_idxs_choice = (0, 12, 25, 37, 49)

    # main calls
    run_spread_analysis(ens_s2s, ordered_idxs_choice)


if __name__ == '__main__':
    main()
