# IceGiantExoplanets

## Literature

ice-giant-exoplanets.pdf is the thesis for the semester project, whith the .tex in the Project Thesis folder

EXO4_GW.pdf is a poster for a conference, with the .tex in the Poster folder

The literature folder contains auxilliary literature

## LISA

binary.py is the main class for the LISA computations: It initialises a DWD + exoplanet system and performes all relevant computations (snr, fisher matrix, relative uncertainties ...) and saves the results for a given system in dict_binaries.txt, as the computations are quite time intensive

binary_test.py is the program where I test the class

uncertainty_plot_Tamanini.py computes the relative uncertainties as in Tamaini (2020) as a check of my methods, tamanini_data.py can plot the result of this paper as a direct comparison

verification_binaries.py computes how many exoplanets can be detected around some verification binaries as specified in [...]

## Ice Giant Mission

ig_binary.py is the main class for the ice giant mission computations, as specified by Armstrong (...) and conceptually similiar to the LISA ones and saved in dict_binaries_ig.py

uncertainty_plot_Tamanini_ig.py computes the relative uncertainties as in Tamaini (2020) for the ice giant mission properties

## LISA x Ice giant

LISA_x_IceGiant.py uses the saved information to compute the uncertainties for a simultaneous search of LISA and an ice giant mission

## Misc.

The rest of the files are mainly taken from Travis Robson et al 2019 Class. Quantum Grav. 36 105011 to compute the strain sensitivity curve or for me to play around with integrations or test my code

I hadn't had a good coding style back when we did the thesis, so best of luck :)
New email for questions: marcus.haberland@aei.mpg.de