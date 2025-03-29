import matplotlib

matplotlib.use('PDF')  # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import datetime
import copy

def plot_with_conf_bounds(ax, record, x_range, label, z_score=1.96, **kwargs):
    avg = record.mean(axis=0)
    stdev = np.std(record, axis=0, ddof=1)
    conf_bound = z_score * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound

    ax.plot(x_range, avg, label=label, **kwargs)
    ax.fill_between(x_range, lower_conf_bound, upper_conf_bound, alpha=0.3, **kwargs)

    return avg[-1], conf_bound[-1]


results_ctl_withneg = [
    {
        'F_q': [-22.34440803527832, -11.863945960998535, -7.353817939758301, -3.0703225135803223, 0.7595230937004089, 3.624729871749878, 6.964207649230957, 9.227322578430176, 10.366497039794922, 11.2396240234375, 12.55330753326416],
        'Grad_variances': [0.010568532161414623, 0.013674727641046047, 0.014491986483335495, 0.015413129702210426, 0.014660436660051346, 0.013513196259737015,0.012697247788310051, 0.011479819193482399, 0.010566779412329197, 0.00997947808355093, 0.009596124291419983]
    },
    {
        'F_q': [-22.369243621826172, -12.604150772094727, -7.903936862945557, -2.379706859588623, 2.0170817375183105, 5.218780994415283, 7.811871528625488, 8.788321495056152, 11.072391510009766, 12.51443099975586, 12.846776008605957],
        'Grad_variances': [0.007775467820465565, 0.014187915250658989, 0.015072896145284176, 0.0152592146769166, 0.01437504030764103, 0.013244781643152237, 0.012141377665102482, 0.012054849416017532, 0.011650425381958485, 0.012385503388941288, 0.011610232293605804]
    },
    {
        'F_q': [-22.74648094177246, -12.679433822631836, -7.767117500305176, -3.346113443374634, 1.2247819900512695, 4.566837310791016, 7.111928939819336, 8.99271297454834, 11.070645332336426, 12.253168106079102, 13.619242668151855,],
        'Grad_variances': [0.009791290387511253, 0.014632932841777802, 0.015243877656757832, 0.015511894598603249, 0.014712698757648468, 0.013994347304105759, 0.013607540167868137, 0.012405809946358204, 0.010968497954308987, 0.010161717422306538, 0.00927541870623827]
    }

]

results_ctl_noneg = [
    {
        'F_q': [-22.34440803527832, -11.404258728027344, -5.957297325134277, 0.3552587628364563, 6.078735828399658, 10.122962951660156, 12.41978931427002, 14.61543083190918, 15.616931915283203, 16.489355087280273, 17.583276748657227],
        'Grad_variances': [0.0106606250628829, 0.010693139396607876, 0.010448543354868889, 0.009005524218082428, 0.006793513894081116, 0.004971492104232311, 0.0038213739171624184, 0.00300543662160635, 0.002438137773424387, 0.0021091182716190815, 0.0018314055632799864],
    },
    {
        'F_q': [-22.369243621826172, -11.673914909362793, -6.72862434387207, 0.8224526643753052, 7.675048351287842, 11.712445259094238, 14.105308532714844, 16.24441146850586, 17.521825790405273, 17.95102882385254, 18.12940788269043],
        'Grad_variances': [0.010042800568044186, 0.0109377671033144, 0.010769001208245754, 0.008991014212369919, 0.006333982106298208, 0.004346050787717104, 0.003236627671867609, 0.002395231043919921, 0.001620782888494432, 0.0013567953137680888, 0.0012946523493155837]
    },
    {
        'F_q': [-22.74648094177246, -11.460172653198242, -5.974493503570557, 0.24315088987350464, 6.915804862976074, 10.767159461975098, 13.467485427856445, 14.94764518737793, 17.616331100463867, 18.117582321166992, 18.348590850830078,],
        'Grad_variances': [0.010578841902315617, 0.011376469396054745, 0.010837068781256676, 0.009090966545045376, 0.006785389501601458, 0.004772842861711979, 0.0033911203499883413, 0.0025191407185047865, 0.00196370342746377, 0.0016393064288422465, 0.0015938804717734456]
    }

]

color_list_for_variances = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light brown', 'xkcd:light lime green', 'xkcd:light navy blue', 'xkcd:light indigo', 'xkcd:olive yellow', 'xkcd:peach', 'xkcd:light lavender', 'xkcd:bright pink' ]
color_list_for_fqs = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black', 'xkcd:brown', 'xkcd:lime green', 'xkcd:navy blue', 'xkcd:indigo', 'xkcd:dark yellow', 'xkcd:dark peach', 'xkcd:lavender', 'xkcd:hot pink']
linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot'] * 5

x_range = np.arange(11)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

f_q_record_ctl_withneg = np.stack([x['F_q'] for x in results_ctl_withneg])
f_q_record_ctl_noneg = np.stack([x['F_q'] for x in results_ctl_noneg])
var_record_ctl_withneg = np.stack([x['Grad_variances'] for x in results_ctl_withneg])
var_record_ctl_noneg = np.stack([x['Grad_variances'] for x in results_ctl_noneg])

plot_with_conf_bounds(
    ax1, f_q_record_ctl_withneg, x_range, label=f"CTL: $E_q[\log (\sigma(s) / q(s))]$",
    color=color_list_for_fqs[0],
    linestyle=linestyle_list[0],
)
plot_with_conf_bounds(
    ax1, f_q_record_ctl_noneg, x_range, label=f"CTL (no second term): $E_q[\log (\sigma(s) / q(s))]$",
    color=color_list_for_fqs[1],
    linestyle=linestyle_list[0],
)
plot_with_conf_bounds(
    ax2, var_record_ctl_withneg, x_range, label=f"CTL: Avg. gradient variance",
    color=color_list_for_variances[0],
    linestyle=linestyle_list[1],
)
# Average across layers of (mean across parameters) variances over time (last 100 steps)
plot_with_conf_bounds(
    ax2, var_record_ctl_noneg, x_range, label=f"CTL (no second term): Avg. gradient variance",
    color=color_list_for_variances[1],
    linestyle=linestyle_list[1],
)
ax1.set_xlabel("Time Step")
ax1.set_ylabel(r"Lower Bound Estimate $(E_q[\log (\sigma(s) / q(s))])$")
ax2.set_ylabel("Average Variance of Gradients over Time")

# Adjust tick colors
# ax1.tick_params(axis="y", colors=color_list_for_fqs[0])
# ax2.tick_params(axis="y", colors=color_list_for_variances[0])

# Combine legends
fig.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.45, 0.5))

# plt.legend()
plt.tight_layout()

figname = "./ctl_nosecondterm_comparison.pdf"
plt.savefig(figname)
