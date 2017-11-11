import main_training

for quantif in [2, 4, 16]:
    for color_mode in ["RGB", "RG"]:
        main_training.main(1, 2, quantif, color_mode)

for quantif in [2, 4, 8, 16]:
    for color_mode in ["RGB", "RG"]:
        main_training.main(1, 9, quantif, color_mode)
