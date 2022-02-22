from ticktack import fitting

fitting.plot_samples(average_path=snakemake.input[0], chains_path=snakemake.input[1:],
                     cbm_models=snakemake.params.cbm_model, cbm_label = snakemake.params.cbm_label,
                     hemisphere=snakemake.params.hemisphere, production_model=snakemake.params.production_model,
                     directory_path="data/" + snakemake.params.event, size=100, size2=30,
                     alpha=0.05, alpha2=0.2, savefig_path=snakemake.output[0], title=snakemake.params.event_label)
