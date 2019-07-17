import json

OUTPUT_FILE = 'manifest2.json'

video_dict = {}

video_dict["segment_duration_ms"] = 2000.0
video_dict["bitrates_kbps"] = [200, 300, 480, 750, 1200, 1850, 2850, 4300, 5300]
video_dict["hd_rewards"] = [1, 2, 3, 4, 8, 12, 15, 20, 30]
video_dict["segment_sizes_bits"] = []

all_sizes = []
for i in range(len(video_dict["bitrates_kbps"])):
    all_sizes.append([])
    with open("video_size_" + str(i)) as f:
        lines = f.readlines()
    for l in lines:
        all_sizes[i].append(int(l))

for j in range(len(all_sizes[0])):
    video_dict["segment_sizes_bits"].append([])
    for k in range(len(video_dict["bitrates_kbps"])):
        video_dict["segment_sizes_bits"][j].append(all_sizes[k][j])

with open(OUTPUT_FILE, 'ab') as out:
    json.dump(video_dict, out)

