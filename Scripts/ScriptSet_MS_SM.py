



for ID in range(NPointings):
    MS=MDC.giveMS(ID)
    exec("MS.uvw=P_%3.3i_uvw"%ID)
    exec("MS.A0=P_%3.3i_A0"%ID)
    exec("MS.A1=P_%3.3i_A1"%ID)
    exec("MS.times_all=P_%3.3i_times_all"%ID)



from PredictDir import ClassHyperH
h=ClassHyperH.ClassHyperH(LRIME,GD)

