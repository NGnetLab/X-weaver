import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = './plot_data/'
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
#SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
SCHEMES = ['scnn', 'fnn']

def main():
        epoch_all = {}
        time_all = {}
        train_loss_all = {}
        train_acc_all = {}
        val_acc0_all = {}
        val_acc1_all = {}
        for scheme in SCHEMES:
                epoch_all[scheme]={}
                time_all[scheme] = {}
                train_loss_all[scheme] = {}
                train_acc_all[scheme] = {}
                val_acc0_all[scheme] = {}
                val_acc1_all[scheme] = {}

        log_files = os.listdir(RESULTS_FOLDER)
        for log_file in log_files:
                epoch = []
                time = []
                train_loss = []
                train_acc = []
                val_acc0 = []
                val_acc1 = []

                print(log_file)

                with open(RESULTS_FOLDER + log_file, 'rb') as f:
                    for line in f:
                        parse = line.split()
                        print(parse)
                        if len(parse) <= 1:
                             break
                        epoch.append(int(parse[0]))
                        time.append(float(parse[4]))
                        train_loss.append(float(parse[1]))
                        train_acc.append(float(parse[2]))
                        val_acc0.append(float(parse[5]))
                        val_acc1.append(float(parse[6]))

                time = np.array(time)
                time -= time[0]
		# print log_file

               
                time_all[scheme] = time
                train_loss_all[scheme] = train_loss
                train_acc_all[scheme] = train_acc
                val_acc0_all[scheme] = val_acc0
                val_acc1_all[scheme] = val_acc1

        fig = plt.figure()
        #fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
 
        figure_val_acc = {}
        figure_val_acc["scnn"] = val_acc0_alli["scnn"]
        figure_val_acc["fnn"] = val_acc1_all["fnn"]
        # ax1 = fig.add_subplot(111)
        for scheme in figure_val_acc:
            ax1.plot(time_all,figure_bitrate[scheme])
        colors = [COLOR_MAP2(i) for i in np.linspace(0, 1, len(ax1.lines))]
        for i, j in enumerate(ax1.lines):
            j.set_color(colors[i])

        SCHEMES_REW = []
        for scheme in figure_bitrate:
            SCHEMES_REW.append(scheme)
        ax1.legend(SCHEMES_REW, loc=4)

        ax1.set_ylabel('Bitrate')
        ax1.set_xlabel('time index')
        plt.show()

         

if __name__ == '__main__':
	main()
