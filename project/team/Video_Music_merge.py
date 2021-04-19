import ffmpeg


input_video = ffmpeg.input("./project/team/data/cartoonize/ONF.avi")
added_audio = ffmpeg.input("./project/team/data/original_Music/ONF.mp3")

print(input_video)
print(added_audio)

(ffmpeg
.concat(input_video, added_audio, v=1, a=1)
.output("./project/team/data/output/ONF.mp4")
.run(cmd=r'C:/Users/AI/Downloads/ffmpeg-4.4-full_build/bin/ffmpeg.exe'))
