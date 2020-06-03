import csv
import os

class CsvRecoder():
    def __init__(self, filepath):
        csvpath = "Saved/" + filepath + "/Record.csv"
        if os.path.exists(csvpath):
            csvfile = open(csvpath, 'a+', newline='')
            self.writer = csv.writer(csvfile)
        else:
            csvfile = open(csvpath, 'w', newline='')
            self.writer = csv.writer(csvfile)
            self.writer.writerow(["Epochs", "Loss", "oRMSE", "oMAE", "vRMSE", "vMAE", "sRMSE", "sMAE"])

    def insertrow(self, epochs, loss, ormse, omae, vrmse, vmae, srmse, smae):
        line = [ "{:.4f}".format(epochs), "{:.4f}".format(loss), "{:.4f}".format(ormse), "{:.4f}".format(omae), "{:.4f}".format(vrmse), "{:.4f}".format(vmae), "{:.4f}".format(srmse), "{:.4f}".format(smae)]
        self.writer.writerow(line)


