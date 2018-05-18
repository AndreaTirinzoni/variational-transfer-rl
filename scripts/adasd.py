ax = plot_ci(100 * _filter, EpRewMean_cb_pois[:, _filter], conf, n_runs, xlabel='trajectories', ylabel='average return', label='CB-POIS', linewidth=2.0, linestyle='-', color='b')
ax = plot_ci(100 * _filter, EpRewMean_pb_pois[:, _filter], conf, n_runs, label='PB-POIS', ax=ax, linewidth=2.0, linestyle=':', color='r')
ax = plot_ci(100 * _filter, EpRewMean_trpo[:, _filter], conf, n_runs, label='TRPO', ax=ax, linewidth=2.0, linestyle='--', color='g')
#ax = plot_ci(100 * _filter, EpRewMean_cb_pois_pdis[:, _filter], conf, n_runs, label='PDIS', ax=ax, linewidth=2.0, linestyle='--', color='y')

#ax.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
ax.xaxis.label.set_color('k')
ax.yaxis.label.set_color('k')
ax.tick_params(axis='x', colors='k', labelsize=TICKS_FONT_SIZE)
ax.tick_params(axis='y', colors='k', labelsize=TICKS_FONT_SIZE)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.tight_layout()
plt.savefig('pdfs/inverted-pendulum_linear.pdf')


figLegend = plt.figure()
legend = plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', fontsize=LEGEND_FONT_SIZE, ncol=2)
legend.get_frame().set_facecolor('none')
legend.get_frame().set_linewidth(0.0)
figLegend.canvas.draw()
bbox  = legend.get_window_extent().transformed(figLegend.dpi_scale_trans.inverted())
figLegend.savefig('pdfs/legend.pdf',
   bbox_inches=bbox)