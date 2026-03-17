input_file = "/home/oleksii/dev/Pixel-Accurate-Epipolar-Guided-Matching/exc/images.txt"
output_file = "/home/oleksii/dev/Pixel-Accurate-Epipolar-Guided-Matching/exc/images_mod.txt"

keep_keywords = ["DSC_0320", "DSC_0321"]

with open(input_file, "r") as f:
    lines = f.readlines()

filtered_lines = [
    line for line in lines
    if any(k in line for k in keep_keywords) or "#"  in line
]

with open(output_file, "w") as f:
    f.writelines(filtered_lines)

print(f"Saved {len(filtered_lines)} lines to {output_file}")