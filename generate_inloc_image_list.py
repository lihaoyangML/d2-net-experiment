import os
relative_path = "../hierarchical_localization/Hierarchical-Localization/datasets/inloc/query/iphone7/"
image_paths = [relative_path + f for f in os.listdir(relative_path) if f.endswith(".JPG")]
with open("image_list_inloc.txt", 'w') as f:
    f.write("\n".join(map(str, image_paths)))