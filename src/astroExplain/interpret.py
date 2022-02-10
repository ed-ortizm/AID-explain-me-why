# save image with positive contributing meaningful features
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    num_features=100,
    hide_rest=False,
    # min_weight = -100
)

galaxyExplanation = mark_boundaries(temp, mask)
# import matplotlib.pyplot as plt
# plt.imshow(galaxyExplanation)
# plt.show()

###############################################################################
save_to = parser.get("directory", "output")
if os.path.exists(save_to) is False:
    os.mkdir(save_to)
np.save(f"{save_to}/{name_galaxy}_allExp.npy", galaxyExplanation)

# heatmap visualization
# code from lime repo
print("Get heat map visualization", end="\n")

# Select the same class explained on the figures above.
ind = explanation.top_labels[0]

# Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

# Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
plt.colorbar()
# plt.show()
plt.savefig(f"{save_to}/{name_galaxy}_heatMap.png")
